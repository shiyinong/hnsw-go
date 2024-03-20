package distance

import (
	"fmt"
)

type Type string

const (
	L2 Type = "L2Distance"
)

var (
	FuncMap = map[Type]func(vec1, vec2 []float32) float32{
		L2: L2Distance,
	}
)

func L2Distance(vec1, vec2 []float32) float32 {
	if len(vec2) != len(vec1) {
		panic(fmt.Sprintf("vec1 dim: [%v] != vec2 dim: [%v]", len(vec1), len(vec2)))
	}
	s := float32(0)
	for i := range vec1 {
		diff := vec1[i] - vec2[i]
		s += diff * diff
	}
	return s
}
