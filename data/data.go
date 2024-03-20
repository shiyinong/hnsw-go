package data

import (
	"math/rand"
	"time"
)

const (
	defaultDim   = 128
	defaultCount = 100000
)

type Doc struct {
	Id     int32
	Vector []float32
}

type Element struct {
	Doc      *Doc
	Distance float32
}

func (e *Element) GetValue() float32 {
	return e.Distance
}

var (
	random = rand.New(rand.NewSource(time.Now().UnixMicro()))
)

func BuildAllDoc(dim, count int32) []*Doc {
	if dim < 1 {
		dim = defaultDim
	}
	if count < 1 {
		count = defaultCount
	}
	Docs := make([]*Doc, count)
	for i := 0; i < len(Docs); i++ {
		Docs[i] = BuildDoc(int32(i), dim)
	}
	return Docs
}

func BuildDoc(id, dim int32) *Doc {
	vector := make([]float32, dim)
	for i := 0; i < len(vector); i++ {
		vector[i] = random.Float32()
	}
	return &Doc{Id: id, Vector: vector}
}
