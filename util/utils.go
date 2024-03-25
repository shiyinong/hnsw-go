package util

import (
	"encoding/binary"
	"io"
)

func Min[T int32 | int64 | int](n1, n2 T) T {
	if n1 < n2 {
		return n1
	}
	return n2
}

func ReadValue[T int | int32 | int64 | float32 | float64](r io.Reader) T {
	var i T
	err := binary.Read(r, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
	return i
}

func WriteValue[T int | int32 | int64 | float32 | float64](v T, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}
