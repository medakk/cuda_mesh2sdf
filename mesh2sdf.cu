// centroid:  [  92.6200991  -157.6624484  -666.61104378]
// scale:  [1.38349843 0.99729681 2.00067234


inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float dot2(float3 a)
{
    return dot(a,a);
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float sign(float x)
{
    float t = x > 0.0;
    return t - (x < 0.0);
}



// inigo quilez: https://www.iquilezles.org/www/articles/triangledistance/triangledistance.htm
__device__
float distance( float3 v1, float3 v2, float3 v3, float3 p )
{
    // prepare data    
    float3 v21 = v2 - v1; float3 p1 = p - v1;
    float3 v32 = v3 - v2; float3 p2 = p - v2;
    float3 v13 = v1 - v3; float3 p3 = p - v3;
    float3 nor = cross( v21, v13 );

    float dist = sqrt( // inside/outside test    
                 ( sign(dot(cross(v21,nor),p1))
                 + sign(dot(cross(v32,nor),p2))
                 + sign(dot(cross(v13,nor),p3))<2.0) 
                  ?
                  // 3 edges    
                  min( min( 
                  dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                  dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                  dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0,1.0)-p3) )
                  :
                  // 1 face    
                  dot(nor,p1)*dot(nor,p1)/dot2(nor) );
    
    // which side of the triangle?
    return sign(dot(nor, p1)) * dist;
}


__global__ void mesh2sdf(float *sdf, int w, int h, int d, float *V, int *F, int nFaces)
{
    const uint y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const uint z = (blockIdx.z * blockDim.z) + threadIdx.z;

    // TODO is this right? (most definitely not)
    if(y >= h || z >= d) {
        return;
    }

    // todo pass in scale dont hardcode 
    const float pt_y = (y - h / 2.0) * 0.99729681 * 64.0 / (float) h;
    const float pt_z = (z - d / 2.0) * 2.00067234 * 64.0 / (float) d;

    for(uint x=0; x<w; x++) {
        const int idx = x + w * (y + d * z);
        float currDist = sdf[idx];

        const float pt_x = (x - w / 2.0) * 1.38349843 * 64.0 / (float) w;
        float3 pt = make_float3(pt_x, pt_y, pt_z);

        for(int f=0; f<nFaces; f++) {
            float3 v1 = make_float3(V[3*F[3*f+0]+0], V[3*F[3*f+0]+1], V[3*F[3*f+0]+2]);
            float3 v2 = make_float3(V[3*F[3*f+1]+0], V[3*F[3*f+1]+1], V[3*F[3*f+1]+2]);
            float3 v3 = make_float3(V[3*F[3*f+2]+0], V[3*F[3*f+2]+1], V[3*F[3*f+2]+2]);

            const float dist = distance(v1, v2, v3, pt);

            if(abs(dist) < abs(currDist)) {
                currDist = dist;
            }
        }

        sdf[idx] = currDist;
    }
}