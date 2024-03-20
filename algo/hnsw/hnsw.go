package hnsw

import (
	"math"
	"math/rand"
	"time"

	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
	"github.com/shiyinong/hnsw-go/util"
)

type Mode int

var (
	Simple    Mode = 0
	Heuristic Mode = 1
)

type HNSW struct {
	// all doc
	docs []*data.Doc
	// size of the dynamic candidate list for insertion
	efCons int
	// size of the dynamic candidate list for search
	ef int
	// friend number of per node at a layer expect layer 0
	m int
	// friend number of per node at layer 0, recommend value: 2*m
	m0 int
	// normalization factor for level generation, recommend value: 1 / lg(m)
	normFactor float64

	mode Mode

	// doc id -> layer id -> neighbor doc
	neighbors  [][][]*neighbor
	entryPoint *data.Doc
	maxLayer   int

	rand *rand.Rand

	disFunc func(vec1, vec2 []float32) float32
}

type neighbor struct {
	doc *data.Doc
	dis float32
}

func BuildHNSW(m, efCons int, mode Mode, disType distance.Type) *HNSW {
	return &HNSW{
		efCons:     efCons,
		m:          m,
		m0:         2 * m,
		normFactor: 1 / math.Log(float64(m)),
		disFunc:    distance.FuncMap[disType],
		rand:       rand.New(rand.NewSource(time.Now().UnixMicro())),
		mode:       mode,
	}
}

func (h *HNSW) Insert(vector []float32) {
	newDoc := &data.Doc{
		Id:     int32(len(h.docs)),
		Vector: vector,
	}
	h.docs = append(h.docs, newDoc)
	maxLayerForNew := int(math.Floor(-math.Log(h.rand.Float64() * h.normFactor)))
	h.neighbors[newDoc.Id] = make([][]*neighbor, maxLayerForNew)
	if h.entryPoint == nil {
		h.entryPoint = newDoc
		return
	}
	entryPoint := h.entryPoint
	for curLayer := h.maxLayer; curLayer > maxLayerForNew; curLayer-- {
		entryPoint = h.searchAtLayerWith1Ef(vector, entryPoint, curLayer)
	}

	for curLayer := maxLayerForNew; curLayer >= 0; curLayer-- {
		docs := h.searchAtLayer(vector, entryPoint, h.efCons, curLayer)
		for _, doc := range docs {
			// add bidirectional connection
			h.connect(newDoc, doc, curLayer)
		}
	}

	if len(h.neighbors[newDoc.Id]) > h.maxLayer {
		h.maxLayer = len(h.neighbors[newDoc.Id])
		h.entryPoint = newDoc
	}
}

func (h *HNSW) searchAtLayer(query []float32, enterPoint *data.Doc, ef, layer int) []*data.Doc {
	visited := make(map[int32]struct{})
	candidates, result := util.NewMinHeap(), util.NewMaxHeap()
	ele := &data.Element{
		Doc:      enterPoint,
		Distance: h.disFunc(query, enterPoint.Vector),
	}
	candidates.Push(ele)
	result.Push(ele)
	for candidates.Size() > 0 {
		candidate := candidates.Pop().(*data.Element)
		if candidate.Distance > result.Top().GetValue() {
			break
		}
		for _, n := range h.neighbors[candidate.Doc.Id][layer] {
			if _, ok := visited[n.doc.Id]; ok {
				continue
			}
			visited[n.doc.Id] = struct{}{}
			newEle := &data.Element{
				Doc:      n.doc,
				Distance: h.disFunc(n.doc.Vector, query),
			}
			if result.Size() < ef {
				result.Push(newEle)
				candidates.Push(newEle)
			} else if result.Top().GetValue() > newEle.Distance {
				result.PopAndPush(newEle)
				candidates.Push(newEle)
			}
		}
	}
	return nil
}

func (h *HNSW) connect(doc1, doc2 *data.Doc, layer int) {
	dis := h.disFunc(doc1.Vector, doc2.Vector)
	// add bidirectional connection
	h.addNeighbor(layer, h.neighbors[doc1.Id][layer], &neighbor{
		doc: doc2,
		dis: dis,
	})
	h.addNeighbor(layer, h.neighbors[doc2.Id][layer], &neighbor{
		doc: doc1,
		dis: dis,
	})
}

func (h *HNSW) searchAtLayerWith1Ef(query []float32, enterPoint *data.Doc, layer int) *data.Doc {
	maxDis := h.disFunc(enterPoint.Vector, query)
	findBetter := false
	for !findBetter {
		for _, n := range h.neighbors[enterPoint.Id][layer] {
			dis := h.disFunc(query, n.doc.Vector)
			if dis < maxDis {
				enterPoint = n.doc
				maxDis = dis
				findBetter = true
			}
		}
	}
	return enterPoint
}

func (h *HNSW) addNeighbor(maxCnt int, neighbors []*neighbor, newNeighbor *neighbor) {
	if h.mode == Simple {
		// h.neighbors[curDoc.Id][layer] is sorted
		neighbors = append(neighbors, newNeighbor)
		idx := len(neighbors) - 1
		for i := len(neighbors) - 2; i >= 0; i-- {
			if newNeighbor.dis > neighbors[i].dis {
				break
			}
			neighbors[i], neighbors[idx] = neighbors[idx], neighbors[i]
			idx = i
		}
		if len(neighbors) > maxCnt {
			neighbors = neighbors[0 : len(neighbors)-1]
		}
	}
	// todo heuristic
}

func (h *HNSW) SearchKNN(query []float32, ef, k int) []*data.Doc {

	return nil
}