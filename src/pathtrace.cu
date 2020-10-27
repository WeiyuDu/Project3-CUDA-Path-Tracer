#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "main.h"
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

#define SORTMATERIAL 0
#define CACHE 0
#define ANTIALISING 0
#define DEPTH_OF_FIELD 0
#define DIRECT_LIGHT 1

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;
        glm::vec3 normal_col = glm::clamp(glm::abs(gBuffer[index].normal) * 255.f, 0.f, 255.f);//(gBuffer[index].normal * 0.5f + glm::vec3(0.5f)) * 255.f;
        glm::vec3 pos_col = glm::clamp(glm::abs(gBuffer[index].pos) * 20.f, 0.f, 255.f);
        pbo[index].w = 0;
        /*
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
        */
        
        // normal
        
        pbo[index].x = normal_col.x;
        pbo[index].y = normal_col.y;
        pbo[index].z = normal_col.z;
        
        
        // pos
        /*
        pbo[index].x = pos_col.x;
        pbo[index].y = pos_col.y;
        pbo[index].z = pos_col.z;
        */
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static Triangle * dev_tris = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static glm::vec3* dev_denoised_in = NULL;
static glm::vec3* dev_denoised_out = NULL;
static glm::vec3* dev_denoised_image = NULL;

static int num_samples = 9;
static float lens_r = 0.4f;
static float f = 10.f;

// TODO: static variables for device memory, any extra info you need, etc
// ...
#if CACHE
static ShadeableIntersection* dev_intersections_cache = NULL;
#endif

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_denoised_in, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_in, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_out, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_out, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));
#if CACHE
    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_tris);
    cudaFree(dev_gBuffer);
    cudaFree(dev_denoised_in);
    cudaFree(dev_denoised_out);
    cudaFree(dev_denoised_image);
#if CACHE
    cudaFree(dev_intersections_cache);
#endif
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if ANTIALISING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 view = cam.view;
        float interp = u01(rng);
        if (cam.move != glm::vec3(0.f)) {
            view = cam.view * (1 - interp) + (cam.view - cam.move) * interp;
        }
        
        segment.ray.direction = glm::normalize(view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );
        
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	const int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
    , Triangle * tris
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) 
            {
                t = meshIntersectionTest(geom, tris, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
#if DIRECT_LIGHT
#else if
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
#endif
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
        pathSegments[idx].remainingBounces--;
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

__global__ void shadeBSDF(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                glm::vec3 intersect = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

                scatterRay(
                    pathSegments[idx],
                    intersect,
                    intersection.surfaceNormal,
                    material,
                    rng);
                
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].pos = pathSegments[idx].ray.origin + gBuffer[idx].t * pathSegments[idx].ray.direction;

    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void finalGatherDenoised(int nPaths, glm::vec3* image, glm::vec3* denoise_out)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        image[index] = denoise_out[index];
    }
}


struct bounce_end {
    __host__ __device__ bool operator()(const PathSegment& seg) {
        return seg.remainingBounces > 0;
    }
};

struct sort_material {
    __host__ __device__ bool operator()(const ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2) {
        return intersect1.materialId > intersect2.materialId;
    }
};

__host__ __device__ glm::vec2 concentricSampleDisk(glm::vec2 u) {
    glm::vec2 uOffset = 2.f * u - glm::vec2(1.f);
    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::vec2(0.f);
    }
    float theta, r;
    float PiOver4 = PI / 4.f;
    float PiOver2 = PI / 2.f;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
        
}

__global__ void updateRayForDepth(Camera cam, float f, float lens_r, PathSegment* pathSegments, int iter)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= cam.resolution.x && y >= cam.resolution.y) {
        return;
    }
    int index = x + (y * cam.resolution.x);
    if (lens_r > 0.f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(-1, 1);

        float rand_x = u01(rng);
        float rand_y = u01(rng);
        glm::vec2 pLens = lens_r * concentricSampleDisk(glm::vec2(rand_x, rand_y));
        
        float ft = glm::abs(f / pathSegments[index].ray.direction.z);
        glm::vec3 pFocus = ft * pathSegments[index].ray.direction + pathSegments[index].ray.origin;

        pathSegments[index].ray.origin += glm::vec3(pLens.x, pLens.y, 0);
        pathSegments[index].ray.direction = glm::normalize(pFocus - pathSegments[index].ray.origin);
    }
    
}

__global__ void atrousConv(Camera cam, glm::vec3* denoised_in, glm::vec3* denoised_out, int iter,
    float sigma_nor, float sigma_col, float sigma_pos,
    GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= cam.resolution.x && y >= cam.resolution.y) {
        return;
    }
    int stepwidth = (int)powf(2.f, iter);
    int index = x + (y * cam.resolution.x);
    glm::vec3 tmp(0.f);
    float coeffs[3]{ 3.f / 8.f, 1.f / 4.f, 1.f / 16.f };
    float normalize = 0.f;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            int new_x = i * stepwidth + x;
            int new_y = j * stepwidth + y;
            float coeff = coeffs[max(abs(i), abs(j))];
            if (new_x < 0 || new_y < 0 || new_x >= cam.resolution.x || new_y >= cam.resolution.y) {
                /*
                if (new_x < 0) {
                    new_x = 0;
                }
                if (new_y < 0) {
                    new_y = 0;
                }
                if (new_x >= cam.resolution.x) {
                    new_x = cam.resolution.x;
                }
                if (new_y >= cam.resolution.y) {
                    new_y = cam.resolution.y;
                }
                */
                continue;
            }
            int new_idx = new_x + (new_y * cam.resolution.x);

            // wn nor
            glm::vec3 diff_nor = gBuffer[new_idx].normal - gBuffer[index].normal;
            float dist_nor = max(glm::dot(diff_nor, diff_nor) / (stepwidth * stepwidth), 0.f);
            float wn = min(exp(-dist_nor / sigma_nor), 1.f);

            // wp pos
            glm::vec3 diff_pos = gBuffer[new_idx].pos - gBuffer[index].pos;
            float dist_pos = glm::dot(diff_pos, diff_pos);
            float wp = min(exp(-dist_pos / sigma_pos), 1.f);

            // wc col
            glm::vec3 diff_col = denoised_in[new_idx] - denoised_in[index];
            float dist_col = glm::dot(diff_col, diff_col);
            float wc = min(exp(-dist_col / sigma_col), 1.f);

            float w = wn * wp * wc * coeff;
            /*
            if (w != 0) {
                w = w * coeff;
            }
            else {
                w = coeff;
            }
            */
            tmp += w * denoised_in[new_idx];
            normalize += w;
        }
    }
    if (normalize > 0) {
        denoised_out[index] = tmp / normalize;
    }
    
}

__global__ void recon(Camera cam, glm::vec3* denoised, glm::vec3* image, glm::vec3* detail) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= cam.resolution.x && y >= cam.resolution.y) {
        return;
    }
    int index = x + (y * cam.resolution.x);
    image[index] = denoised[index] + detail[index];
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
 //void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    //const int traceDepth = 16;
    //std::cout << traceDepth << std::endl;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#if DEPTH_OF_FIELD
    updateRayForDepth << <blocksPerGrid2d, blockSize2d >> > (cam, f, lens_r, dev_paths, iter);
    checkCUDAError("depth of field");
#endif
	int depth = 0;
    int ctr = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    int num_paths_orig = num_paths;
    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    const dim3 numblocksDenoise(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    while (!iterationComplete) {
        //cout << depth << endl;
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE
        if (iter <= 1 && depth == 0) {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_tris
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections_cache, dev_intersections, 
                pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth == 0) {
            cudaMemcpy(dev_intersections, dev_intersections_cache,
                pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                , dev_tris
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
#else
	    // tracing
	    computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		    depth
		    , num_paths
		    , dev_paths
		    , dev_geoms
		    , hst_scene->geoms.size()
		    , dev_intersections
            , dev_tris
		    );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#endif
        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

	    depth++;
	    // --- Shading Stage ---
	    // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if SORTMATERIAL
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections+num_paths, dev_paths, sort_material());
#endif

        shadeBSDF << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_path_end, bounce_end());
        if (dev_path_end == dev_paths) {
            iterationComplete = true;
        }
        num_paths = dev_path_end - dev_paths;

        
	}
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths_orig, dev_image, dev_paths);
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    if (ui_denoise) {
        float avg_nor = 0.f;
        float avg_col = 0.f;
        float avg_pos = 0.f;
        float sigma_nor = ui_normalWeight;
        float sigma_col = ui_colorWeight;// / powf(2.f, iter);
        float sigma_pos = ui_positionWeight;
        /*
        cout << pixelcount << endl;
        for (int i = 0; i < pixelcount; i++) {
            avg_nor += glm::l2Norm(dev_gBuffer[i].normal);
            avg_pos += glm::l2Norm(dev_gBuffer[i].pos);
           //avg_col += glm::dot(dev_image[i], dev_image[i]);
           // cout << i << endl;
        }
        
        
        cout << avg_nor << " " << avg_pos << endl;
        avg_nor = avg_nor / num_paths_orig;
        avg_pos = avg_pos / num_paths_orig;
        //avg_col = avg_col / (float)pixelcount;
        float sigma_nor = 1.f;
        float sigma_col = 1.f;
        float sigma_pos = 1.f;
        cout << avg_nor << " " << avg_pos << endl;
        for (int i = 0; i < num_paths_orig; i++) {
           // avg_nor += sqrt(glm::dot(dev_gBuffer[i].normal, dev_gBuffer[i].normal));
            //avg_pos += sqrt(glm::dot(dev_gBuffer[i].pos, dev_gBuffer[i].pos));
            //sigma_nor += powf(sqrt(glm::dot(dev_gBuffer[i].normal, dev_gBuffer[i].normal)) - avg_nor, 2.f);
            //sigma_pos += powf(sqrt(glm::dot(dev_gBuffer[i].pos, dev_gBuffer[i].pos)) - avg_pos, 2.f);
            //sigma_col += glm::dot(dev_image[i], dev_image[i]) - avg_col;
        }
        
        sigma_nor /= (float)num_paths_orig;
        sigma_pos /= (float)num_paths_orig;
        //sigma_col /= (float)pixelcount;
        
        float sigma_nor = 1.f;
        float sigma_col = 1.f;
        float sigma_pos = 1.f;
        */
        cudaMemcpy(dev_denoised_in, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
        int num_s = (int)glm::log2(ui_filterSize / 5) + 1;
        for (int i = 0; i <= num_s; i++) {
            atrousConv << <numblocksDenoise, blockSize2d >> > (cam, dev_denoised_in, dev_denoised_out, i,
                sigma_nor, sigma_col, sigma_pos,
                dev_gBuffer);
            sigma_col = sigma_col / 0.5f;
            //sigma_nor = sigma_nor / 0.5f;
            swap(dev_denoised_in, dev_denoised_out);
        }

        finalGatherDenoised << <numBlocksPixels, blockSize1d >> > (num_paths_orig, dev_denoised_image, dev_denoised_in);
        // Retrieve image from GPU
        cudaMemcpy(hst_scene->state.denoised_image.data(), dev_denoised_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }

    checkCUDAError("pathtrace");
}

void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    if (ui_denoise) {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image);
    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }
    
}