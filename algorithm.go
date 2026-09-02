// Copyright (c) Meta Platforms, Inc. and affiliates.
// SPDX-License-Identifier: BSD-3-Clause
//
// This file is a Go port of the PDQ hashing and downscaling implementation in
// Meta's ThreatExchange repository, pinned at commit
// 07b82cb6e87b7e0ac7fc2a01d865df5db10ee1f2.

package pdq

import "context"

func computeJaroszWindow(dimension int) int {
	return (dimension + 2*downsampleSize - 1) / (2 * downsampleSize)
}

func jaroszFilter(
	ctx context.Context,
	buffer1 []float32,
	buffer2 []float32,
	rows int,
	columns int,
	rowWindow int,
	columnWindow int,
) error {
	const passes = 2
	for range passes {
		if err := boxAlongRows(ctx, buffer1, buffer2, rows, columns, rowWindow); err != nil {
			return err
		}
		if err := boxAlongColumns(ctx, buffer2, buffer1, rows, columns, columnWindow); err != nil {
			return err
		}
	}
	return nil
}

func boxAlongRows(
	ctx context.Context,
	input, output []float32,
	rows, columns, window int,
) error {
	for row := range rows {
		if row&63 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		box1D(input, row*columns, output, row*columns, columns, 1, window)
	}
	return nil
}

func boxAlongColumns(
	ctx context.Context,
	input, output []float32,
	rows, columns, window int,
) error {
	for column := range columns {
		if column&63 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		box1D(input, column, output, column, rows, columns, window)
	}
	return nil
}

func box1D(
	input []float32,
	inputOffset int,
	output []float32,
	outputOffset int,
	length int,
	stride int,
	window int,
) {
	halfWindow := (window + 2) / 2
	phase1 := halfWindow - 1
	phase2 := window - halfWindow + 1
	phase3 := length - window
	phase4 := halfWindow - 1

	left, right, outputIndex := 0, 0, 0
	var sum float32
	currentWindow := 0

	for range phase1 {
		sum += input[inputOffset+right]
		currentWindow++
		right += stride
	}
	for range phase2 {
		sum += input[inputOffset+right]
		currentWindow++
		output[outputOffset+outputIndex] = sum / float32(currentWindow)
		right += stride
		outputIndex += stride
	}
	for range phase3 {
		sum += input[inputOffset+right]
		sum -= input[inputOffset+left]
		output[outputOffset+outputIndex] = sum / float32(currentWindow)
		left += stride
		right += stride
		outputIndex += stride
	}
	for range phase4 {
		sum -= input[inputOffset+left]
		currentWindow--
		output[outputOffset+outputIndex] = sum / float32(currentWindow)
		left += stride
		outputIndex += stride
	}
}

func decimate(
	input []float32,
	inputRows int,
	inputColumns int,
	output *[downsampleSize * downsampleSize]float32,
) {
	for outputRow := range downsampleSize {
		inputRow := int((float64(outputRow) + 0.5) * float64(inputRows) / downsampleSize)
		for outputColumn := range downsampleSize {
			inputColumn := int(
				(float64(outputColumn) + 0.5) * float64(inputColumns) / downsampleSize,
			)
			output[outputRow*downsampleSize+outputColumn] = input[inputRow*inputColumns+inputColumn]
		}
	}
}

func imageQuality(input *[downsampleSize * downsampleSize]float32) uint8 {
	gradientSum := 0
	for row := range downsampleSize - 1 {
		for column := range downsampleSize {
			difference := int(
				(input[row*downsampleSize+column] - input[(row+1)*downsampleSize+column]) *
					100 / 255,
			)
			if difference < 0 {
				difference = -difference
			}
			gradientSum += difference
		}
	}
	for row := range downsampleSize {
		for column := range downsampleSize - 1 {
			difference := int(
				(input[row*downsampleSize+column] - input[row*downsampleSize+column+1]) *
					100 / 255,
			)
			if difference < 0 {
				difference = -difference
			}
			gradientSum += difference
		}
	}

	return uint8(min(gradientSum/90, 100))
}

func (h *Hasher) dct64To16(
	input *[downsampleSize * downsampleSize]float32,
	output *[dctSize * dctSize]float32,
) {
	var intermediate [dctSize * downsampleSize]float32
	for i := range dctSize {
		for j := range downsampleSize {
			var sum float32
			for k := range downsampleSize {
				sum += h.dct[i*downsampleSize+k] * input[k*downsampleSize+j]
			}
			intermediate[i*downsampleSize+j] = sum
		}
	}

	for i := range dctSize {
		for j := range dctSize {
			var sum float32
			for k := range downsampleSize {
				sum += intermediate[i*downsampleSize+k] * h.dct[j*downsampleSize+k]
			}
			output[i*dctSize+j] = sum
		}
	}
}

func transformDCT(
	input *[dctSize * dctSize]float32,
	output *[dctSize * dctSize]float32,
	transform int,
) {
	for i := range dctSize {
		for j := range dctSize {
			value := input[i*dctSize+j]
			switch transform {
			case 1: // Counterclockwise 90 degrees.
				if j&1 == 0 {
					value = -value
				}
				output[j*dctSize+i] = value
			case 2: // 180 degrees.
				if (i+j)&1 != 0 {
					value = -value
				}
				output[i*dctSize+j] = value
			case 3: // Counterclockwise 270 degrees.
				if i&1 == 0 {
					value = -value
				}
				output[j*dctSize+i] = value
			case 4: // Flip X.
				if i&1 == 0 {
					value = -value
				}
				output[i*dctSize+j] = value
			case 5: // Flip Y.
				if j&1 == 0 {
					value = -value
				}
				output[i*dctSize+j] = value
			case 6: // Reflect across the positive diagonal.
				output[j*dctSize+i] = value
			case 7: // Reflect across the negative diagonal.
				if (i+j)&1 != 0 {
					value = -value
				}
				output[j*dctSize+i] = value
			}
		}
	}
}

func hashDCT(input *[dctSize * dctSize]float32) [HashBits / 8]byte {
	median := torbenMedian(input[:])
	var words [HashBits / 16]uint16
	for bit, value := range input {
		if value > median {
			words[bit/16] |= 1 << (bit % 16)
		}
	}

	var hash [HashBits / 8]byte
	for index := range words {
		word := words[len(words)-1-index]
		hash[index*2] = byte(word >> 8)
		hash[index*2+1] = byte(word & 0xff)
	}
	return hash
}

func torbenMedian(values []float32) float32 {
	minimum, maximum := values[0], values[0]
	for _, value := range values[1:] {
		if value < minimum {
			minimum = value
		}
		if value > maximum {
			maximum = value
		}
	}

	middle := (len(values) + 1) / 2
	var guess, maximumBelowGuess, minimumAboveGuess float32
	var less, equal, greater int
	for {
		guess = (minimum + maximum) / 2
		less, equal, greater = 0, 0, 0
		maximumBelowGuess = minimum
		minimumAboveGuess = maximum

		for _, value := range values {
			switch {
			case value < guess:
				less++
				if value > maximumBelowGuess {
					maximumBelowGuess = value
				}
			case value > guess:
				greater++
				if value < minimumAboveGuess {
					minimumAboveGuess = value
				}
			default:
				equal++
			}
		}

		if less <= middle && greater <= middle {
			break
		}
		if less > greater {
			maximum = maximumBelowGuess
		} else {
			minimum = minimumAboveGuess
		}
	}

	switch {
	case less >= middle:
		return maximumBelowGuess
	case less+equal >= middle:
		return guess
	default:
		return minimumAboveGuess
	}
}
