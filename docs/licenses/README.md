# Licenses

- The repository retains upstream license files (e.g.
  `apps/commerce-java/LICENSE`, Apache License 2.0).
- `THIRD-PARTY-LICENSES.md` summarizes the license posture of the main runtime
  components.
- `third_party/SBOM` records component versions and licenses at the current
  revision; regenerate it before release.

Langfuse: core is MIT, `ee/` is separately licensed — keep the enterprise
directory out of this repository and record upstream revisions in
`third_party/UPSTREAMS.md`.
