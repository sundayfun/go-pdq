# go-pdq

`go-pdq` is a pure Go implementation of Meta's 256-bit PDQ perceptual image
hash for static JPEG and PNG images. It returns the original hash, seven
dihedral transforms, and PDQ's 0-100 image-quality score.

The implementation is ported from Meta's ThreatExchange PDQ code at commit
`07b82cb6e87b7e0ac7fc2a01d865df5db10ee1f2`. Compatibility tests use hashes
produced by that pinned implementation.

## Install

```bash
go get github.com/sundayfun/go-pdq
```

## Usage

```go
hasher := pdq.New()
result, err := hasher.Hash(ctx, encodedJPEGOrPNG)
if err != nil {
	return err
}

original := result.Hashes[0]
quality := result.Quality
```

`Hashes[0]` is the original orientation. The remaining hashes cover rotations
and reflections. Persisting only `Hashes[0]` and comparing it against all eight
query hashes supports right-angle rotations and reflections without storing
eight fingerprints.

Quality measures usable visual structure; it is not a similarity score. Match
and quality thresholds are application policy and intentionally remain outside
this package.

Before hashing, inputs with a dimension larger than 512 pixels are resized
proportionally to match Meta's file adapters. PNG alpha is ignored: the encoded
RGB values are hashed without compositing onto a background.

## Verification

```bash
go test -race ./...
go vet ./...
```

The test suite covers exact PNG compatibility, JPEG tolerance, large-image
normalization, alpha handling, image quality, and all eight dihedral hashes.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
