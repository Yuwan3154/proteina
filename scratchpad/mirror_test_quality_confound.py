"""Is the t_beta "significant" mirror result a STRUCTURE-QUALITY ARTIFACT?

val/is_mirrored fires only when proper RMSD > 2x the reflection-allowed RMSD AND the gap exceeds 1 A.
That criterion is a RATIO, so it needs the reflection to be markedly better than the proper
superposition. When a structure is poorly formed, both RMSDs are large and dominated by general
inaccuracy rather than by handedness, the ratio collapses toward 1, and the test cannot fire no
matter how mirrored the fold is.

If that is happening, a lower-quality arm shows a LOWER is_mirrored rate for reasons having nothing
to do with chirality -- and the apparent "significant" result is an artifact.
"""

# (label, rmsd_proper, rmsd_reflected) at matched step 3143
ROWS = [
    ("control @3143", 7.394, 4.802),
    ("t_beta  @3143", 28.065, 25.311),
    ("t_beta  @2643", 88.526, 86.429),
    ("t_beta  @2143", 129.748, 128.123),
]
THRESHOLD = 2.0   # val/is_mirrored requires proper > 2 x reflected

print(f"{'round':>15} {'proper':>9} {'refl':>9} {'ratio':>7}  can the 2x test fire?")
for label, pr, rf in ROWS:
    ratio = pr / rf
    can = "YES" if ratio > THRESHOLD else "NO — mechanically cannot"
    print(f"{label:>15} {pr:9.3f} {rf:9.3f} {ratio:7.2f}  {can}")

print(f"\nThe aggregate ratio is below {THRESHOLD} in EVERY row, so these are round-level means and")
print("individual chains still vary -- but the ordering is the point:")
print("  control ratio 1.54  vs  t_beta ratio 1.11")
print("\n=> The worse the structures, the closer the ratio sits to 1, and the LESS OFTEN a")
print("   ratio-based mirror test can fire. t_beta's structures are 3.8x worse at this step")
print("   (28.1 A vs 7.4 A), so its lower is_mirrored rate is at least PARTLY a quality artifact.")
print("\n⛔ The native-free CA-dihedral detector does not use the native at all, so it is the more")
print("   trustworthy instrument here -- and it says 3/16 (18.8%), NOT significant vs the control.")
print("\n⛔ CONCLUSION: the 'significant at 5%' GT result must NOT be reported as evidence that")
print("   high-noise training reduces mirroring. The verdict remains NOT READABLE.")
