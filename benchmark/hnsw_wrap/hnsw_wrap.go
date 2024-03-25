package hnsw_wrap

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/shiyinong/hnsw-go/algo/hnsw"
	"github.com/shiyinong/hnsw-go/algo/nsw"
	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
	"github.com/shiyinong/hnsw-go/util"
)

type HnswWrap struct {
	Hnsw     *hnsw.HNSW
	Nsw      *nsw.NSW
	TestData []*data.Doc
	TopK     [][]*data.Doc
}

func SaveHnswWrap(wrap *HnswWrap, path string) {
	start := time.Now()
	file, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	h := wrap.Hnsw
	writer := bufio.NewWriter(file)
	util.WriteValue[int32](h.EfCons, writer)
	util.WriteValue[int32](h.Ef, writer)
	util.WriteValue[int32](h.M, writer)
	util.WriteValue[int32](h.M0, writer)
	util.WriteValue[float64](h.NormFactor, writer)
	util.WriteValue[int32](int32(h.Mode), writer)
	util.WriteValue[int32](h.EntryPoint.Id, writer)
	util.WriteValue[int32](h.MaxLayer, writer)
	util.WriteValue[int32](int32(h.DisType), writer)
	util.WriteValue[int32](int32(len(h.Docs)), writer)
	util.WriteValue[int32](int32(len(h.Docs[0].Vector)), writer)
	for _, doc := range h.Docs {
		util.WriteValue[int32](doc.Id, writer)
		for _, v := range doc.Vector {
			util.WriteValue[float32](v, writer)
		}
	}
	for _, layers := range h.Neighbors {
		util.WriteValue[int32](int32(len(layers)), writer)
		for _, layer := range layers {
			util.WriteValue[int32](int32(len(layer)), writer)
			for _, n := range layer {
				util.WriteValue[int32](n.Doc.Id, writer)
				util.WriteValue[float32](n.Dis, writer)
			}
		}
	}

	util.WriteValue[int32](wrap.Nsw.F, writer)
	for _, link := range wrap.Nsw.Links {
		util.WriteValue[int32](int32(len(link)), writer)
		for _, v := range link {
			util.WriteValue[int32](v, writer)
		}
	}

	util.WriteValue[int32](int32(len(wrap.TestData)), writer)
	util.WriteValue[int32](int32(len(wrap.TestData[0].Vector)), writer)
	for _, doc := range wrap.TestData {
		util.WriteValue[int32](doc.Id, writer)
		for _, v := range doc.Vector {
			util.WriteValue[float32](v, writer)
		}
	}
	for _, res := range wrap.TopK {
		util.WriteValue[int32](int32(len(res)), writer)
		for _, v := range res {
			util.WriteValue[int32](v.Id, writer)
		}
	}

	if err = writer.Flush(); err != nil {
		panic(err)
	}

	if err = file.Close(); err != nil {
		panic(err)
	}
	fmt.Printf("saveHnswWrap hnsw index cost: [%v]\n", time.Since(start))
}

func LoadHnswWrap(path string) *HnswWrap {
	start := time.Now()
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	reader := bufio.NewReader(file)
	efCons := util.ReadValue[int32](reader)
	ef := util.ReadValue[int32](reader)
	m := util.ReadValue[int32](reader)
	m0 := util.ReadValue[int32](reader)
	normFactor := util.ReadValue[float64](reader)
	mode := util.ReadValue[int32](reader)
	entryPointId := util.ReadValue[int32](reader)
	maxLayer := util.ReadValue[int32](reader)
	disType := util.ReadValue[int32](reader)
	docSize := util.ReadValue[int32](reader)
	d := util.ReadValue[int32](reader)
	docs := make([]*data.Doc, docSize)
	for i := int32(0); i < docSize; i++ {
		vector := make([]float32, d)
		id := util.ReadValue[int32](reader)
		for j := int32(0); j < d; j++ {
			vector[j] = util.ReadValue[float32](reader)
		}
		docs[i] = &data.Doc{
			Id:     id,
			Vector: vector,
		}
	}
	neighbors := make([][][]*hnsw.Neighbor, docSize)
	for i := int32(0); i < docSize; i++ {
		layerCnt := util.ReadValue[int32](reader)
		layers := make([][]*hnsw.Neighbor, layerCnt)
		for j := int32(0); j < layerCnt; j++ {
			neighborCnt := util.ReadValue[int32](reader)
			ns := make([]*hnsw.Neighbor, neighborCnt)
			for n := int32(0); n < neighborCnt; n++ {
				id := util.ReadValue[int32](reader)
				dis := util.ReadValue[float32](reader)
				ns[n] = &hnsw.Neighbor{
					Doc: docs[id],
					Dis: dis,
				}
			}
			layers[j] = ns
		}
		neighbors[i] = layers
	}

	nswF := util.ReadValue[int32](reader)
	nswLinks := make([][]int32, docSize)
	for i := int32(0); i < docSize; i++ {
		size := util.ReadValue[int32](reader)
		link := make([]int32, size)
		for j := int32(0); j < size; j++ {
			link[j] = util.ReadValue[int32](reader)
		}
		nswLinks[i] = link
	}

	testDataSize := util.ReadValue[int32](reader)
	d = util.ReadValue[int32](reader)
	testData := make([]*data.Doc, testDataSize)
	for i := int32(0); i < testDataSize; i++ {
		vector := make([]float32, d)
		id := util.ReadValue[int32](reader)
		for j := int32(0); j < d; j++ {
			vector[j] = util.ReadValue[float32](reader)
		}
		testData[i] = &data.Doc{
			Id:     id,
			Vector: vector,
		}
	}
	topK := make([][]*data.Doc, testDataSize)
	for i := int32(0); i < testDataSize; i++ {
		size := util.ReadValue[int32](reader)
		arr := make([]*data.Doc, size)
		for j := int32(0); j < size; j++ {
			arr[j] = docs[util.ReadValue[int32](reader)]
		}
		topK[i] = arr
	}

	err = file.Close()
	if err != nil {
		panic(err)
	}
	fmt.Printf("load hnsw index cost: [%v]\n", time.Since(start))

	return &HnswWrap{
		Hnsw: &hnsw.HNSW{
			Docs:       docs,
			EfCons:     efCons,
			Ef:         ef,
			M:          m,
			M0:         m0,
			NormFactor: normFactor,
			Mode:       hnsw.Mode(mode),
			Neighbors:  neighbors,
			EntryPoint: docs[entryPointId],
			MaxLayer:   maxLayer,
			Rand:       rand.New(rand.NewSource(time.Now().UnixNano())),
			DisType:    distance.Type(disType),
			DisFunc:    distance.FuncMap[distance.Type(disType)],
		},
		Nsw: &nsw.NSW{
			Docs:    docs,
			Links:   nswLinks,
			F:       nswF,
			DisFunc: distance.FuncMap[distance.Type(disType)],
		},
		TestData: testData,
		TopK:     topK,
	}
}
