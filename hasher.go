// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: BSD-3-Clause
//
// Package pdq implements Meta's PDQ perceptual image hash for static images.
package pdq

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"

	xdraw "golang.org/x/image/draw"
)

const (
	HashBits       = 256
	DihedralHashes = 8
	maxPixels      = 16_000_000
	minDimension   = 5
	// Meta's Python and C++ file adapters cap large inputs before hashing.
	maxHashDimension = 512

	dctSize        = 16
	downsampleSize = 64
)

// Result contains the original PDQ fingerprint followed by its seven
// dihedral transforms, plus the algorithm's 0-100 image quality score.
// Quality describes how much usable visual structure the image contains; it
// is not a similarity score. Callers persist Hashes[0] and may query with all
// eight hashes to recognize right-angle rotations and reflections.
type Result struct {
	Hashes  [DihedralHashes][HashBits / 8]byte
	Quality uint8
}

// Hasher holds the immutable DCT coefficients shared by all hash operations.
// It is safe for concurrent use.
type Hasher struct {
	dct [dctSize * downsampleSize]float32
}

// New constructs a PDQ hasher. The implementation is ported from Meta's
// ThreatExchange PDQ implementation pinned at commit
// 07b82cb6e87b7e0ac7fc2a01d865df5db10ee1f2.
func New() *Hasher {
	const pi = 3.14159265358979323846
	scale := float32(math.Sqrt(2.0 / downsampleSize))

	hasher := &Hasher{}
	for i := range dctSize {
		for j := range downsampleSize {
			hasher.dct[i*downsampleSize+j] = float32(
				float64(scale) * math.Cos((pi/2/downsampleSize)*float64(i+1)*float64(2*j+1)),
			)
		}
	}
	return hasher
}

// Hash computes the original and seven dihedral PDQ hashes for a static JPEG
// or PNG. PDQ is defined on luminance, so alpha is deliberately ignored when
// RGB values are read from a PNG; no background colour is introduced.
func (h *Hasher) Hash(ctx context.Context, imageBytes []byte) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, fmt.Errorf("pdq: %w", err)
	}

	config, format, err := image.DecodeConfig(bytes.NewReader(imageBytes))
	if err != nil {
		return Result{}, fmt.Errorf("pdq: decode image header: %w", err)
	}
	if format != "jpeg" && format != "png" {
		return Result{}, fmt.Errorf("pdq: unsupported image format %q", format)
	}
	if err := validateDimensions(config.Width, config.Height); err != nil {
		return Result{}, err
	}

	decoded, decodedFormat, err := image.Decode(bytes.NewReader(imageBytes))
	if err != nil {
		return Result{}, fmt.Errorf("pdq: decode %s: %w", format, err)
	}
	if decodedFormat != format {
		return Result{}, fmt.Errorf("pdq: decoded format %q differs from header %q", decodedFormat, format)
	}
	decoded, err = resizeInput(ctx, decoded)
	if err != nil {
		return Result{}, err
	}

	bounds := decoded.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if err := validateDimensions(width, height); err != nil {
		return Result{}, err
	}
	if width < minDimension || height < minDimension {
		return Result{}, nil
	}

	buffer1 := make([]float32, width*height)
	if err := fillLuma(ctx, decoded, buffer1); err != nil {
		return Result{}, err
	}
	buffer2 := make([]float32, len(buffer1))

	if err := jaroszFilter(
		ctx,
		buffer1,
		buffer2,
		height,
		width,
		computeJaroszWindow(width),
		computeJaroszWindow(height),
	); err != nil {
		return Result{}, fmt.Errorf("pdq: %w", err)
	}

	var downsampled [downsampleSize * downsampleSize]float32
	decimate(buffer1, height, width, &downsampled)

	var dctOutput [dctSize * dctSize]float32
	h.dct64To16(&downsampled, &dctOutput)

	result := Result{Quality: imageQuality(&downsampled)}
	result.Hashes[0] = hashDCT(&dctOutput)

	var transformed [dctSize * dctSize]float32
	for transform := 1; transform < DihedralHashes; transform++ {
		transformDCT(&dctOutput, &transformed, transform)
		result.Hashes[transform] = hashDCT(&transformed)
	}
	return result, nil
}

func resizeInput(ctx context.Context, source image.Image) (image.Image, error) {
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	targetWidth, targetHeight := hashDimensions(width, height)
	if targetWidth == width && targetHeight == height {
		return source, nil
	}

	if opaque, ok := source.(interface{ Opaque() bool }); !ok || !opaque.Opaque() {
		opaqueSource := image.NewNRGBA(image.Rect(0, 0, width, height))
		for y := range height {
			if err := checkRowContext(ctx, y); err != nil {
				return nil, err
			}
			for x := range width {
				red, green, blue := unassociatedRGB(source.At(bounds.Min.X+x, bounds.Min.Y+y))
				opaqueSource.SetNRGBA(x, y, color.NRGBA{
					R: red,
					G: green,
					B: blue,
					A: 0xff,
				})
			}
		}
		source = opaqueSource
	}

	resized := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	xdraw.CatmullRom.Scale(resized, resized.Bounds(), source, source.Bounds(), xdraw.Src, nil)
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("pdq: %w", err)
	}
	return resized, nil
}

func hashDimensions(width, height int) (int, int) {
	largest := max(width, height)
	if largest <= maxHashDimension {
		return width, height
	}

	return max(1, (width*maxHashDimension+largest/2)/largest),
		max(1, (height*maxHashDimension+largest/2)/largest)
}

func validateDimensions(width, height int) error {
	if width <= 0 || height <= 0 || width > maxPixels/height {
		return fmt.Errorf(
			"pdq: image dimensions %dx%d exceed %d pixels",
			width,
			height,
			maxPixels,
		)
	}
	return nil
}

func fillLuma(ctx context.Context, source image.Image, destination []float32) error {
	bounds := source.Bounds()
	width := bounds.Dx()

	switch source := source.(type) {
	case *image.NRGBA:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if err := checkRowContext(ctx, y); err != nil {
				return err
			}
			pixel := source.PixOffset(bounds.Min.X, y)
			row := (y - bounds.Min.Y) * width
			for x := range width {
				destination[row+x] = luminance(
					source.Pix[pixel],
					source.Pix[pixel+1],
					source.Pix[pixel+2],
				)
				pixel += 4
			}
		}
		return nil
	case *image.RGBA:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if err := checkRowContext(ctx, y); err != nil {
				return err
			}
			pixel := source.PixOffset(bounds.Min.X, y)
			row := (y - bounds.Min.Y) * width
			for x := range width {
				r, g, b := unassociatedRGBA(
					source.Pix[pixel],
					source.Pix[pixel+1],
					source.Pix[pixel+2],
					source.Pix[pixel+3],
				)
				destination[row+x] = luminance(r, g, b)
				pixel += 4
			}
		}
		return nil
	case *image.YCbCr:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if err := checkRowContext(ctx, y); err != nil {
				return err
			}
			row := (y - bounds.Min.Y) * width
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				value := source.YCbCrAt(x, y)
				r, g, b := color.YCbCrToRGB(value.Y, value.Cb, value.Cr)
				destination[row+x-bounds.Min.X] = luminance(r, g, b)
			}
		}
		return nil
	case *image.Gray:
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if err := checkRowContext(ctx, y); err != nil {
				return err
			}
			pixel := source.PixOffset(bounds.Min.X, y)
			row := (y - bounds.Min.Y) * width
			for x := range width {
				destination[row+x] = float32(source.Pix[pixel])
				pixel++
			}
		}
		return nil
	case *image.Paletted:
		paletteLuma := make([]float32, len(source.Palette))
		for index, value := range source.Palette {
			r, g, b := unassociatedRGB(value)
			paletteLuma[index] = luminance(r, g, b)
		}
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			if err := checkRowContext(ctx, y); err != nil {
				return err
			}
			pixel := source.PixOffset(bounds.Min.X, y)
			row := (y - bounds.Min.Y) * width
			for x := range width {
				destination[row+x] = paletteLuma[source.Pix[pixel]]
				pixel++
			}
		}
		return nil
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		if err := checkRowContext(ctx, y); err != nil {
			return err
		}
		row := (y - bounds.Min.Y) * width
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b := unassociatedRGB(source.At(x, y))
			destination[row+x-bounds.Min.X] = luminance(r, g, b)
		}
	}
	return nil
}

func checkRowContext(ctx context.Context, row int) error {
	if row&63 != 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("pdq: %w", err)
	}
	return nil
}

func luminance(red, green, blue uint8) float32 {
	return 0.299*float32(red) + 0.587*float32(green) + 0.114*float32(blue)
}

func unassociatedRGBA(red, green, blue, alpha uint8) (uint8, uint8, uint8) {
	if alpha == 0 {
		return 0, 0, 0
	}
	if alpha == 0xff {
		return red, green, blue
	}
	return unpremultiply8(red, alpha),
		unpremultiply8(green, alpha),
		unpremultiply8(blue, alpha)
}

func unassociatedRGBA64(red, green, blue, alpha uint16) (uint8, uint8, uint8) {
	if alpha == 0 {
		return 0, 0, 0
	}
	if alpha == 0xffff {
		return uint8(red >> 8), uint8(green >> 8), uint8(blue >> 8)
	}
	return unpremultiply16(red, alpha),
		unpremultiply16(green, alpha),
		unpremultiply16(blue, alpha)
}

func unpremultiply8(value, alpha uint8) uint8 {
	scaled := uint16(value) * 0xff / uint16(alpha)
	if scaled > 0xff {
		return 0xff
	}
	return uint8(scaled)
}

func unpremultiply16(value, alpha uint16) uint8 {
	scaled := uint32(value) * 0xffff / uint32(alpha) >> 8
	if scaled > 0xff {
		return 0xff
	}
	return uint8(scaled)
}

// unassociatedRGB returns colour channels without compositing a background.
// image/png decodes alpha images as NRGBA, so this also preserves their RGB
// values when alpha is zero, matching Pillow's RGBA-to-RGB conversion used by
// Meta's reference Python implementation.
func unassociatedRGB(value color.Color) (uint8, uint8, uint8) {
	switch value := value.(type) {
	case color.NRGBA:
		return value.R, value.G, value.B
	case color.RGBA:
		return unassociatedRGBA(value.R, value.G, value.B, value.A)
	case color.NRGBA64:
		return uint8(value.R >> 8), uint8(value.G >> 8), uint8(value.B >> 8)
	case color.RGBA64:
		return unassociatedRGBA64(value.R, value.G, value.B, value.A)
	case color.Gray:
		return value.Y, value.Y, value.Y
	case color.Gray16:
		grey := uint8(value.Y >> 8)
		return grey, grey, grey
	case color.YCbCr:
		return color.YCbCrToRGB(value.Y, value.Cb, value.Cr)
	}

	r, g, b, a := value.RGBA()
	if a == 0 {
		return 0, 0, 0
	}
	return unpremultiplyColor(r, a), unpremultiplyColor(g, a), unpremultiplyColor(b, a)
}

func unpremultiplyColor(value, alpha uint32) uint8 {
	if alpha != 0xffff {
		value = uint32(min(uint64(value)*0xffff/uint64(alpha), uint64(0xffff)))
	}
	return uint8(value >> 8)
}
