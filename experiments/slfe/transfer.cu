/* Distributed under the Boost Software License, Version 1.0.
 * Experimental GPU-only conservative DG transport on periodic affine triangles.
 * NB and the mean-orthonormal basis are injected by the Python launch layer.
 */
struct Point
{
  double x, y;
};

__device__ inline Point sub(Point a, Point b)
{
  return {a.x-b.x, a.y-b.y};
}

__device__ inline double cross(Point a, Point b)
{
  return a.x*b.y-a.y*b.x;
}

__device__ inline Point map(const double* t, double r, double s)
{
  return {t[0]+r*(t[2]-t[0])+s*(t[4]-t[0]),
          t[1]+r*(t[3]-t[1])+s*(t[5]-t[1])};
}

__device__ inline Point inverse(const double* t, Point p)
{
  Point e = {t[2]-t[0], t[3]-t[1]};
  Point f = {t[4]-t[0], t[5]-t[1]};
  Point a = {p.x-t[0], p.y-t[1]};
  double d = cross(e, f);
  return {cross(a, f)/d, cross(e, a)/d};
}

__device__ inline double determinant(const double* t)
{
  return (t[2]-t[0])*(t[5]-t[1])-(t[4]-t[0])*(t[3]-t[1]);
}

__device__ inline int modulo(int i, int n)
{
  int r = i % n;
  return r < 0 ? r+n : r;
}

__device__ inline void target_triangle(const double* triangles, int n,
  int ix, int iy, int half, double* target, int* id)
{
  int mx = modulo(ix, n), my = modulo(iy, n);
  *id = 2*(my*n+mx)+half;
  double sx = (ix-mx)/n, sy = (iy-my)/n;
  for (int k=0; k<3; ++k)
  {
    target[2*k] = triangles[6*(*id)+2*k]+sx;
    target[2*k+1] = triangles[6*(*id)+2*k+1]+sy;
  }
}

__device__ inline double signed_edge(Point a, Point b, Point p)
{
  return cross(sub(b, a), sub(p, a));
}

// Clip in source reference coordinates: all geometric tolerances are then
// independent of the physical mesh size. Convex triangle intersection has <=6
// vertices; twelve slots accommodate duplicated touching endpoints transiently.
__device__ inline int intersection(const double* moved, const double* target, Point* poly)
{
  poly[0] = {0., 0.}; poly[1] = {1., 0.}; poly[2] = {0., 1.};
  Point edges[3];
  for (int k=0; k<3; ++k)
    edges[k] = inverse(moved, {target[2*k], target[2*k+1]});
  int count = 3;
  for (int edge=0; edge<3 && count; ++edge)
  {
    Point a = edges[edge], b = edges[(edge+1)%3], temp[12];
    int next = 0;
    Point previous = poly[count-1];
    double dp = signed_edge(a, b, previous);
    for (int k=0; k<count; ++k)
    {
      Point current = poly[k];
      double dc = signed_edge(a, b, current);
      bool inp = dp >= 0., inc = dc >= 0.;
      if (inp != inc)
      {
        double t = dp/(dp-dc);
        temp[next++] = {previous.x+t*(current.x-previous.x),
                        previous.y+t*(current.y-previous.y)};
      }
      if (inc)
        temp[next++] = current;
      previous = current;
      dp = dc;
    }
    count = next;
    for (int k=0; k<count; ++k)
      poly[k] = temp[k];
  }
  return count;
}

extern "C" __global__ void build_blocks(
  const double* triangles, const double* moved, int n, int first, int cells,
  int slots, const double* quadrature, int nq, int mode,
  int* destinations, double* blocks, int* errors)
{
  int pair = blockIdx.x*blockDim.x+threadIdx.x;
  if (pair >= cells*slots)
    return;
  int source = first+pair/slots, candidate = pair % slots;
  destinations[pair] = -1;
  const double* t = moved+6*source;
  double det = determinant(t);
  if (!(det > 0.))
  {
    atomicAdd(errors, 1);
    return;
  }
  // Vertex displacement is bounded by 0.24/n in the test mesh. One extra
  // logical square on each side covers all candidate target triangles.
  int xmin = (int)floor(fmin(t[0], fmin(t[2], t[4]))*n)-1;
  int xmax = (int)floor(fmax(t[0], fmax(t[2], t[4]))*n)+1;
  int ymin = (int)floor(fmin(t[1], fmin(t[3], t[5]))*n)-1;
  int ymax = (int)floor(fmax(t[1], fmax(t[3], t[5]))*n)+1;
  int width = xmax-xmin+1, height = ymax-ymin+1;
  if (2*width*height > slots)
  {
    if (candidate == 0)
      atomicAdd(errors+1, 1);
    return;
  }
  if (candidate >= 2*width*height)
    return;
  int ix = xmin+(candidate/2)%width;
  int iy = ymin+(candidate/2)/width;
  double target[6];
  int destination;
  target_triangle(triangles, n, ix, iy, candidate%2, target, &destination);
  // Reject disjoint axis-aligned boxes before clipping or quadrature.
  if (fmax(t[0], fmax(t[2], t[4])) < fmin(target[0], fmin(target[2], target[4])) ||
      fmin(t[0], fmin(t[2], t[4])) > fmax(target[0], fmax(target[2], target[4])) ||
      fmax(t[1], fmax(t[3], t[5])) < fmin(target[1], fmin(target[3], target[5])) ||
      fmin(t[1], fmin(t[3], t[5])) > fmax(target[1], fmax(target[3], target[5])))
    return;
  double matrix[NB*NB];
  for (int k=0; k<NB*NB; ++k)
    matrix[k] = 0.;
  double ratio = determinant(triangles+6*source)/determinant(target);
  if (mode == 0)
  {
    Point poly[12];
    int count = intersection(t, target, poly);
    for (int fan=1; fan<count-1; ++fan)
    {
      Point e = sub(poly[fan], poly[0]), f = sub(poly[fan+1], poly[0]);
      double d = cross(e, f);
      if (!(d > 0.))
        continue;
      for (int q=0; q<nq; ++q)
      {
        double r = quadrature[3*q], s = quadrature[3*q+1];
        Point a = {poly[0].x+r*e.x+s*f.x, poly[0].y+r*e.y+s*f.y};
        Point z = inverse(target, map(t, a.x, a.y));
        double source_basis[NB], target_basis[NB];
        basis(a.x, a.y, source_basis);
        basis(z.x, z.y, target_basis);
        double w = 2.*ratio*d*quadrature[3*q+2];
        for (int i=0; i<NB; ++i)
          for (int j=0; j<NB; ++j)
            matrix[i*NB+j] += w*target_basis[i]*source_basis[j];
      }
    }
  }
  else
  {
    // Source quadrature deposition baseline. Resolve boundary ties with the
    // same deterministic first-containing-cell convention as the point locator.
    for (int q=0; q<nq; ++q)
    {
      double r = quadrature[3*q], s = quadrature[3*q+1];
      Point z = inverse(target, map(t, r, s));
      const double face_tolerance = 2.e-12;
      if (z.x < -face_tolerance || z.y < -face_tolerance || z.x+z.y > 1.+face_tolerance)
        continue;
      if (z.x < face_tolerance || z.y < face_tolerance || 1.-z.x-z.y < face_tolerance)
      {
        Point x = map(t, r, s);
        int gx = (int)floor(x.x*n), gy = (int)floor(x.y*n), owner = -1;
        for (int y=gy-1; y<=gy+1 && owner<0; ++y)
          for (int xx=gx-1; xx<=gx+1 && owner<0; ++xx)
            for (int h=0; h<2 && owner<0; ++h)
            {
              double other[6];
              int id;
              target_triangle(triangles, n, xx, y, h, other, &id);
              Point u = inverse(other, x);
              if (u.x >= -face_tolerance && u.y >= -face_tolerance && u.x+u.y <= 1.+face_tolerance)
                owner = id;
            }
        if (owner != destination)
          continue;
      }
      double source_basis[NB], target_basis[NB];
      basis(r, s, source_basis);
      basis(z.x, z.y, target_basis);
      double w = 2.*ratio*quadrature[3*q+2];
      for (int i=0; i<NB; ++i)
        for (int j=0; j<NB; ++j)
          matrix[i*NB+j] += w*target_basis[i]*source_basis[j];
    }
  }
  if (!(matrix[0] > 0.))
    return;
  destinations[pair] = destination;
  for (int k=0; k<NB*NB; ++k)
    blocks[pair*NB*NB+k] = matrix[k];
}

extern "C" __global__ void apply_blocks(const int* source, const int* destination,
  const double* blocks, int pairs, const double* input, double* output, int fields)
{
  int item = blockIdx.x*blockDim.x+threadIdx.x;
  if (item >= pairs*fields*NB)
    return;
  int i = item % NB, field = (item/NB)%fields, pair = item/(NB*fields);
  double value = 0.;
  for (int j=0; j<NB; ++j)
    value += blocks[pair*NB*NB+i*NB+j]*input[(source[pair]*fields+field)*NB+j];
  atomicAdd(output+(destination[pair]*fields+field)*NB+i, value);
}

extern "C" __global__ void apply_blocks_transpose(const int* source, const int* destination,
  const double* blocks, int pairs, const double* input, double* output, int fields)
{
  int item = blockIdx.x*blockDim.x+threadIdx.x;
  if (item >= pairs*fields*NB)
    return;
  int j = item % NB, field = (item/NB)%fields, pair = item/(NB*fields);
  double value = 0.;
  for (int i=0; i<NB; ++i)
    value += blocks[pair*NB*NB+i*NB+j]*input[(destination[pair]*fields+field)*NB+i];
  atomicAdd(output+(source[pair]*fields+field)*NB+j, value);
}

extern "C" __global__ void evaluate_basis(const double* points, int count, double* values)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < count)
    basis(points[2*i], points[2*i+1], values+NB*i);
}

extern "C" __global__ void locate_points(const double* triangles, int n,
  const double* points, int count, int* destinations, double* references, int* errors)
{
  int k = blockIdx.x*blockDim.x+threadIdx.x;
  if (k >= count)
    return;
  Point x = {points[2*k], points[2*k+1]};
  int ix = (int)floor(x.x*n), iy = (int)floor(x.y*n);
  const double face_tolerance = 2.e-12;
  for (int y=iy-1; y<=iy+1; ++y)
    for (int xx=ix-1; xx<=ix+1; ++xx)
      for (int half=0; half<2; ++half)
      {
        int cell;
        double target[6];
        target_triangle(triangles, n, xx, y, half, target, &cell);
        Point z = inverse(target, x);
        if (z.x >= -face_tolerance && z.y >= -face_tolerance && z.x+z.y <= 1.+face_tolerance)
        {
          destinations[k] = cell;
          references[2*k] = z.x;
          references[2*k+1] = z.y;
          return;
        }
      }
  destinations[k] = -1;
  atomicAdd(errors, 1);
}

extern "C" __global__ void weighted_particles(const int* source, const int* target,
  const double* sb, const double* tb, const double* weights, const double* areas,
  int count, const double* input, double* output, int fields)
{
  int item = blockIdx.x*blockDim.x+threadIdx.x;
  if (item >= count*fields)
    return;
  int particle = item/fields, field = item%fields;
  int a = source[particle], b = target[particle];
  double value = 0.;
  for (int j=0; j<NB; ++j)
    value += sb[particle*NB+j]*input[(a*fields+field)*NB+j];
  value *= weights[particle]/areas[b];
  for (int i=0; i<NB; ++i)
    atomicAdd(output+(b*fields+field)*NB+i, value*tb[particle*NB+i]);
}

extern "C" __global__ void deposit_particles(
  const double* triangles, const double* moved, int n,
  const double* quadrature, int nq, const double* input, double* output,
  int fields, int* errors)
{
  int particle = blockIdx.x*blockDim.x+threadIdx.x;
  int count = 2*n*n*nq;
  if (particle >= count)
    return;
  int cell = particle/nq, q = particle%nq;
  double r = quadrature[3*q], s = quadrature[3*q+1];
  Point x = map(moved+6*cell, r, s);
  int ix = (int)floor(x.x*n), iy = (int)floor(x.y*n);
  double target[6], tb[NB], sb[NB];
  int destination = -1;
  // Bounded jitter of the topology makes a 3x3 square stencil sufficient.
  // Deterministic first-containing-cell convention resolves points on faces.
  const double face_tolerance = 2.e-12;
  for (int y=iy-1; y<=iy+1 && destination<0; ++y)
    for (int xx=ix-1; xx<=ix+1 && destination<0; ++xx)
      for (int half=0; half<2 && destination<0; ++half)
      {
        int candidate;
        target_triangle(triangles, n, xx, y, half, target, &candidate);
        Point z = inverse(target, x);
        if (z.x >= -face_tolerance && z.y >= -face_tolerance && z.x+z.y <= 1.+face_tolerance)
        {
          destination = candidate;
          basis(z.x, z.y, tb);
        }
      }
  if (destination < 0)
  {
    atomicAdd(errors, 1);
    return;
  }
  basis(r, s, sb);
  double w = 2.*quadrature[3*q+2]*determinant(triangles+6*cell)/determinant(target);
  for (int field=0; field<fields; ++field)
  {
    double value = 0.;
    for (int j=0; j<NB; ++j)
      value += input[(cell*fields+field)*NB+j]*sb[j];
    for (int i=0; i<NB; ++i)
      atomicAdd(output+(destination*fields+field)*NB+i, w*value*tb[i]);
  }
}
