// -------------------------------------------------------------------------
//    Participating media
// -------------------------------------------------------------------------
// A participating medium inside an axis-aligned box, ported from pbrt-v4
// (media.h / scattering.h) but using RGB coefficients instead of spectra.
//
// Two modes:
//  - Homogeneous (pbrt's HomogeneousMedium): analytic free-flight sampling and
//    transmittance. A channel is picked at random and the result is weighted by
//    the average pdf over the RGB channels (one-sample MIS over channels).
//  - Density grid (pbrt's GridMedium): delta tracking for free-flight sampling and
//    ratio tracking for transmittance. A coarse majorant grid is traversed with a 3D DDA
//    (pbrt's DDAMajorantIterator) so empty regions are skipped quickly. RGB is handled with
//    spectral tracking (Kutz et al. 2017): collision probabilities use channel averages
//    and the throughput is reweighted per channel.
//
// Requires: gData, densityGrid, majorantGrid, rand(), RngStateType, PI, FLT_MAX.

#ifndef MEDIUM_H
#define MEDIUM_H

#define MEDIUM_PASSED 0     //< Ray left the medium segment without a real collision
#define MEDIUM_SCATTERED 1  //< Ray scattered inside the medium
#define MEDIUM_ABSORBED 2   //< Ray was absorbed inside the medium

// Safety limits so a single ray can never hang the GPU
#define MEDIUM_MAX_DDA_STEPS 1024
#define MEDIUM_MAX_COLLISIONS 1024

// -------------------------------------------------------------------------
//    Geometry and phase function
// -------------------------------------------------------------------------

// Ray vs. axis-aligned box (slab test). Returns the overlap [t0, t1] clipped to [0, tMax].
bool intersectBox(float3 origin, float3 dir, float tMax, float3 boxMin, float3 boxMax, out float t0, out float t1)
{
	float3 invDir = 1.0f / dir;
	float3 tA = (boxMin - origin) * invDir;
	float3 tB = (boxMax - origin) * invDir;
	float3 tNear = min(tA, tB);
	float3 tFar = max(tA, tB);

	t0 = max(max(tNear.x, tNear.y), max(tNear.z, 0.0f));
	t1 = min(min(tFar.x, tFar.y), min(tFar.z, tMax));
	return t0 < t1;
}

// Henyey-Greenstein phase function. cosTheta is measured between the ray's
// propagation direction and the scattered direction, so g > 0 means forward scattering.
float hgPhase(float g, float cosTheta)
{
	float denom = 1.0f + g * g - 2.0f * g * cosTheta;
	return (1.0f - g * g) / (4.0f * PI * denom * sqrt(max(denom, 1e-7f)));
}

// Orthonormal basis around n (Duff et al. 2017)
void mediumBasis(float3 n, out float3 t, out float3 b)
{
	float s = n.z >= 0.0f ? 1.0f : -1.0f;
	float a = -1.0f / (s + n.z);
	float c = n.x * n.y * a;
	t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
	b = float3(c, s + n.y * n.y * a, -n.y);
}

// Samples a new direction from the HG phase function around the propagation direction 'dir'.
// The sample is drawn exactly proportional to the phase function, so its weight is 1.
float3 sampleHG(float g, float3 dir, float2 u)
{
	float cosTheta;
	if (abs(g) < 1e-3f) {
		cosTheta = 1.0f - 2.0f * u.x;
	} else {
		float sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * u.x);
		cosTheta = (1.0f + g * g - sq * sq) / (2.0f * g);
	}
	cosTheta = clamp(cosTheta, -1.0f, 1.0f);
	float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	float phi = TWO_PI * u.y;

	float3 t, b;
	mediumBasis(dir, t, b);
	return normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * dir);
}

// -------------------------------------------------------------------------
//    Homogeneous medium
// -------------------------------------------------------------------------

// Base coefficients (before multiplying by the density grid)
float3 getMediumSigmaA() { return gData.mediumSigmaA * gData.mediumDensity; }
float3 getMediumSigmaS() { return gData.mediumSigmaS * gData.mediumDensity; }
float3 getMediumSigmaT() { return getMediumSigmaA() + getMediumSigmaS(); }

// Samples a collision inside the medium segment [t0, t1].
// Returns true if a scattering event happened at distance t, false if the ray passed through.
// In both cases, 'weight' is the factor to multiply the path throughput with.
bool sampleMediumDistance(float3 sigma_t, float3 sigma_s, float t0, float t1, float uChannel, float uDistance,
	out float t, out float3 weight)
{
	int channel = min(int(uChannel * 3.0f), 2);
	float sigmaC = sigma_t[channel];
	t = (sigmaC > 0.0f) ? t0 - log(1.0f - uDistance) / sigmaC : t1;

	if (t < t1) {
		// Scattering event: f = sigma_s * Tr, pdf = average over channels of sigma_t * Tr
		float3 Tr = exp(-sigma_t * (t - t0));
		float pdf = dot(sigma_t * Tr, (1.0f / 3.0f).xxx);
		weight = sigma_s * Tr / pdf;
		return true;
	}

	// Passed through: f = Tr, probability = average over channels of Tr
	t = t1;
	float3 Tr = exp(-sigma_t * (t1 - t0));
	float pdf = dot(Tr, (1.0f / 3.0f).xxx);
	weight = Tr / pdf;
	return false;
}

// -------------------------------------------------------------------------
//    Density grid medium
// -------------------------------------------------------------------------

// Density at normalized box coordinates uvw in [0,1]^3. Trilinear interpolation between
// voxel centers with clamp-to-edge, done by hand so it matches the majorant grid exactly.
float lookupDensity(float3 uvw)
{
	uint3 dim;
	densityGrid.GetDimensions(dim.x, dim.y, dim.z);

	float3 g = uvw * float3(dim) - 0.5f;
	int3 i0 = int3(floor(g));
	float3 f = g - float3(i0);
	int3 maxIndex = int3(dim) - 1;
	int3 a = clamp(i0, 0, maxIndex);
	int3 b = clamp(i0 + 1, 0, maxIndex);

	float d000 = densityGrid.Load(int4(a.x, a.y, a.z, 0));
	float d100 = densityGrid.Load(int4(b.x, a.y, a.z, 0));
	float d010 = densityGrid.Load(int4(a.x, b.y, a.z, 0));
	float d110 = densityGrid.Load(int4(b.x, b.y, a.z, 0));
	float d001 = densityGrid.Load(int4(a.x, a.y, b.z, 0));
	float d101 = densityGrid.Load(int4(b.x, a.y, b.z, 0));
	float d011 = densityGrid.Load(int4(a.x, b.y, b.z, 0));
	float d111 = densityGrid.Load(int4(b.x, b.y, b.z, 0));

	float d00 = lerp(d000, d100, f.x);
	float d10 = lerp(d010, d110, f.x);
	float d01 = lerp(d001, d101, f.x);
	float d11 = lerp(d011, d111, f.x);
	return lerp(lerp(d00, d10, f.y), lerp(d01, d11, f.y), f.z);
}

// Walks the segment [t0, t1] through the majorant grid with a 3D DDA and runs
//  - delta tracking (ratioTracking == false): returns the first real collision, or MEDIUM_PASSED.
//    'weight' is multiplied by the spectral tracking weights and tScatter receives the collision distance.
//  - ratio tracking (ratioTracking == true): never stops at collisions, 'weight' is multiplied
//    by the transmittance estimate. Returns MEDIUM_ABSORBED if Russian roulette killed the estimate.
uint trackGridMedium(float3 origin, float3 dir, float t0, float t1, bool ratioTracking,
	inout RngStateType rngState, inout float3 weight, out float tScatter)
{
	tScatter = t1;

	float3 sigmaA = getMediumSigmaA();
	float3 sigmaS = getMediumSigmaS();
	float3 sigmaT = sigmaA + sigmaS;
	float sigmaTMax = max(sigmaT.x, max(sigmaT.y, sigmaT.z));
	if (sigmaTMax <= 0.0f) return MEDIUM_PASSED;

	float3 boxMin = gData.mediumBoundsMin;
	float3 extent = gData.mediumBoundsMax - gData.mediumBoundsMin;

	// Ray in majorant grid space [0, res]^3 (t stays the same since the mapping is affine)
	uint3 res;
	majorantGrid.GetDimensions(res.x, res.y, res.z);
	float3 gridOrigin = (origin - boxMin) / extent * float3(res);
	float3 gridDir = dir / extent * float3(res);

	float3 entry = gridOrigin + gridDir * t0;
	int3 cell = clamp(int3(floor(entry)), 0, int3(res) - 1);
	int3 stepDir = (gridDir >= 0.0f) ? int3(1, 1, 1) : int3(-1, -1, -1);

	// Parametric distance to the next cell boundary on each axis, and between boundaries
	float3 tNext;
	float3 tDelta;
	[unroll] for (int axis = 0; axis < 3; axis++) {
		if (abs(gridDir[axis]) < 1e-12f) {
			tNext[axis] = FLT_MAX;
			tDelta[axis] = FLT_MAX;
		} else {
			float boundary = float(cell[axis] + (gridDir[axis] > 0.0f ? 1 : 0));
			tNext[axis] = (boundary - gridOrigin[axis]) / gridDir[axis];
			tDelta[axis] = abs(1.0f / gridDir[axis]);
		}
	}

	float tCur = t0;
	int collisions = 0;

	[loop] for (int ddaStep = 0; ddaStep < MEDIUM_MAX_DDA_STEPS; ddaStep++) {

		float tCellEnd = min(min(tNext.x, tNext.y), min(tNext.z, t1));
		float sigmaMaj = sigmaTMax * majorantGrid.Load(int4(cell, 0));

		// Delta / ratio tracking inside this cell with a constant majorant
		if (sigmaMaj > 0.0f) {
			[loop] while (collisions < MEDIUM_MAX_COLLISIONS) {
				float t = tCur - log(1.0f - rand(rngState)) / sigmaMaj;
				if (t >= tCellEnd) break;
				tCur = t;
				collisions++;

				float3 uvw = (origin + dir * t - boxMin) / extent;
				float density = lookupDensity(uvw);
				float3 sA = sigmaA * density;
				float3 sS = sigmaS * density;
				float3 sN = max(sigmaMaj - (sA + sS), 0.0f);

				if (ratioTracking) {
					weight *= sN / sigmaMaj;

					// Russian roulette once the transmittance estimate gets small
					float maxWeight = max(weight.x, max(weight.y, weight.z));
					if (maxWeight < 0.1f) {
						if (rand(rngState) < 0.5f) { weight = 0.0f; return MEDIUM_ABSORBED; }
						weight *= 2.0f;
					}
					continue;
				}

				// Spectral tracking: event probabilities from channel averages
				float pA = dot(sA, (1.0f / 3.0f).xxx) / sigmaMaj;
				float pS = dot(sS, (1.0f / 3.0f).xxx) / sigmaMaj;
				float pN = 1.0f - pA - pS;
				float u = rand(rngState);

				if (u < pA) {
					// Absorption: no emission, so the path contributes nothing more
					return MEDIUM_ABSORBED;
				}
				if (u < pA + pS) {
					weight *= sS / (sigmaMaj * pS);
					tScatter = t;
					return MEDIUM_SCATTERED;
				}
				if (pN > 0.0f) weight *= sN / (sigmaMaj * pN);
			}
		}

		if (tCellEnd >= t1) break;
		tCur = tCellEnd;

		// Step into the neighboring cell along the axis with the closest boundary
		if (tNext.x <= tNext.y && tNext.x <= tNext.z) {
			cell.x += stepDir.x;
			tNext.x += tDelta.x;
		} else if (tNext.y <= tNext.z) {
			cell.y += stepDir.y;
			tNext.y += tDelta.y;
		} else {
			cell.z += stepDir.z;
			tNext.z += tDelta.z;
		}
		if (any(cell < 0) || any(cell >= int3(res))) break;
	}

	return MEDIUM_PASSED;
}

// -------------------------------------------------------------------------
//    Entry points used by the path tracer
// -------------------------------------------------------------------------

// Samples the medium along [t0, t1]. Updates throughput and returns one of MEDIUM_PASSED,
// MEDIUM_SCATTERED (tScatter is the collision distance) or MEDIUM_ABSORBED.
uint sampleMediumVolumeRay(float3 origin, float3 dir, float t0, float t1, inout RngStateType rngState,
	inout float3 throughput, out float tScatter)
{
	if (gData.mediumUseGrid)
		return trackGridMedium(origin, dir, t0, t1, false, rngState, throughput, tScatter);

	float3 weight;
	bool scattered = sampleMediumDistance(getMediumSigmaT(), getMediumSigmaS(), t0, t1,
		rand(rngState), rand(rngState), tScatter, weight);
	throughput *= weight;
	return scattered ? MEDIUM_SCATTERED : MEDIUM_PASSED;
}

// Transmittance through the medium box along a segment (used for shadow rays)
float3 mediumTransmittance(float3 origin, float3 dir, float tMax, inout RngStateType rngState)
{
	if (!gData.enableMedium) return 1.0f.xxx;

	float t0, t1;
	if (!intersectBox(origin, dir, tMax, gData.mediumBoundsMin, gData.mediumBoundsMax, t0, t1)) return 1.0f.xxx;

	if (gData.mediumUseGrid) {
		float3 Tr = 1.0f.xxx;
		float unused;
		trackGridMedium(origin, dir, t0, t1, true, rngState, Tr, unused);
		return Tr;
	}

	return exp(-getMediumSigmaT() * (t1 - t0));
}

#endif // MEDIUM_H
