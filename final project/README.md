# From Chaos to Cosmos: Analyzing Galaxy Formation in h277

This project analyzes 13 billion years of galaxy evolution using the h277 cosmological simulation.

## What We Did

We tracked how a chaotic gas cloud transforms into a mature spiral galaxy by analyzing five snapshots spanning 9 billion years. Think of it like watching time-lapse photography of the universe, except each "frame" took billions of years.

**Key findings:**
- Stellar mass doubled from 2.0 × 10¹⁰ to 4.1 × 10¹⁰ solar masses
- The galaxy evolved from an irregular blob into a proper Milky Way-like spiral
- Rotation curve stays flat at ~220 km/s, matching real observations
- Dark matter completely dominates the outer regions

## The Analysis

We used **pynbody** to calculate mass profiles, rotation curves, and surface densities across different cosmic epochs. Then we manually verified all the physics from first principles (just Newton's laws and some careful math) to make sure we actually understood what was happening.

The trickiest part? Centering the galaxy correctly. Pro tip: center on stars, not dark matter.

## Why This Matters

You can't just watch a galaxy form in real life—it takes billions of years. Simulations like h277 let us test if our understanding of physics is correct by starting with basic initial conditions and seeing if we get something that looks like the real universe.

Spoiler: we do. The simulation produces a galaxy that matches Milky Way observations almost perfectly.

---

**Course:** PHYS 75800 - Galactic Physics I  
**Professor:** Dr. Charlotte Welker  
**December 2025**
