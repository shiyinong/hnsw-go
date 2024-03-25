package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
	"github.com/shiyinong/hnsw-go/util"
)

type Mode int32

var (
	Simple    Mode = 0
	Heuristic Mode = 1
)

type HNSW struct {
	// all Doc
	Docs []*data.Doc
	// size of the dynamic candidate list for insertion
	EfCons int32
	// size of the dynamic candidate list for search
	Ef int32
	// friend number of per node at a layer expect layer 0
	M int32
	// friend number of per node at layer 0, recommend value: 2*M
	M0 int32
	// normalization factor for level generation, recommend value: 1 / lg(M)
	NormFactor float64

	Mode Mode

	// Doc id -> layer id -> Neighbor Doc
	Neighbors  [][][]*Neighbor
	EntryPoint *data.Doc
	MaxLayer   int32

	Rand *rand.Rand

	DisType distance.Type
	DisFunc func(vec1, vec2 []float32) float32

	ComputeCnt int64
}

type Neighbor struct {
	Doc *data.Doc
	Dis float32
}

func (h *HNSW) Stat() {
	fmt.Printf("HNSW params:\nM: [%v], M0: [%v], EfCons: [%v], Mode: [%v], EF: [%v], NormFactor: [%.2f], DataSize: [%v], Dim: [%v]\n",
		h.M, h.M0, h.EfCons, h.Mode, h.Ef, h.NormFactor, len(h.Docs), len(h.Docs[0].Vector))
	arr := make([]int, h.MaxLayer+1)
	connCnt := make([]int, h.MaxLayer+1)
	for _, n := range h.Neighbors {
		arr[len(n)-1]++
		for i, v := range n {
			connCnt[i] += len(v)
		}
	}
	cnt := 0
	for i := len(arr) - 1; i >= 0; i-- {
		cnt += arr[i]
		fmt.Printf("layer id: [%v],\tnode count: [%v],\tavg neighbors count: [%v]\n", i, cnt, connCnt[i]/cnt)
	}
}

func BuildHNSW(m, efCons int32, mode Mode, disType distance.Type) *HNSW {
	return &HNSW{
		EfCons:     efCons,
		M:          m,
		M0:         2 * m,
		NormFactor: 1 / math.Log(float64(m)),
		DisType:    disType,
		DisFunc:    distance.FuncMap[disType],
		Rand:       rand.New(rand.NewSource(time.Now().UnixMicro())),
		Mode:       mode,
	}
}

func (h *HNSW) Insert(newDoc *data.Doc) {
	h.Docs = append(h.Docs, newDoc)
	maxLayerForNew := int32(math.Floor(-math.Log(h.Rand.Float64()) * h.NormFactor))
	h.Neighbors = append(h.Neighbors, make([][]*Neighbor, maxLayerForNew+1))
	if h.EntryPoint == nil {
		h.MaxLayer = maxLayerForNew
		h.EntryPoint = newDoc
		return
	}
	entryPoint := h.EntryPoint
	for curLayer := h.MaxLayer; curLayer > maxLayerForNew; curLayer-- {
		entryPoint = h.searchAtLayerWith1Ef(newDoc.Vector, entryPoint, curLayer)
	}

	for curLayer := util.Min(maxLayerForNew, h.MaxLayer); curLayer >= 0; curLayer-- {
		maxHeap := h.searchAtLayer(newDoc.Vector, entryPoint, h.EfCons, curLayer)
		h.Neighbors[newDoc.Id][curLayer] = h.selectNeighborsFromMaxHeap(maxHeap, h.M)
		for _, neighbor := range h.Neighbors[newDoc.Id][curLayer] {
			h.Neighbors[neighbor.Doc.Id][curLayer] = h.addNeighbor(
				h.Neighbors[neighbor.Doc.Id][curLayer],
				&Neighbor{
					Doc: newDoc,
					Dis: neighbor.Dis,
				},
				curLayer,
			)
		}
		if len(h.Neighbors[newDoc.Id][curLayer]) > 0 {
			entryPoint = h.Neighbors[newDoc.Id][curLayer][0].Doc
		}
	}

	if maxLayerForNew > h.MaxLayer {
		h.MaxLayer = maxLayerForNew
		h.EntryPoint = newDoc
	}
}

func (h *HNSW) selectHeuristicNeighborsFromMinHeap(minHeap *util.Heap, maxCnt int32) []*Neighbor {
	selected, discard := util.NewMinHeap(), util.NewMinHeap()
	neighbors := []*Neighbor{}
	for minHeap.Size() > 0 && selected.Size() < int(maxCnt) {
		cur := minHeap.Pop().(*data.Element)
		flag := true
		for _, element := range selected.Elements {
			if cur.Distance > h.DisFunc(cur.Doc.Vector, element.(*data.Element).Doc.Vector) {
				flag = false
				break
			}
		}
		if flag {
			selected.Push(cur)
		} else {
			discard.Push(cur)
		}
	}
	for selected.Size() < int(maxCnt) && discard.Size() > 0 {
		selected.Push(discard.Pop())
	}
	for selected.Size() > 0 {
		ele := selected.Pop().(*data.Element)
		neighbors = append(neighbors, &Neighbor{
			Doc: ele.Doc,
			Dis: ele.Distance,
		})
	}
	return neighbors
}

func (h *HNSW) selectNeighborsFromMaxHeap(maxHeap *util.Heap, maxCnt int32) []*Neighbor {
	if h.Mode == Simple {
		for maxHeap.Size() > int(maxCnt) {
			maxHeap.Pop()
		}
		neighbors := make([]*Neighbor, maxHeap.Size())
		for i := len(neighbors) - 1; i >= 0; i-- {
			ele := maxHeap.Pop().(*data.Element)
			neighbors[i] = &Neighbor{
				Doc: ele.Doc,
				Dis: ele.Distance,
			}
		}
		return neighbors
	} // else h.Mode == Heuristic
	minHeap := util.NewMinHeap()
	for maxHeap.Size() > 0 {
		minHeap.Push(maxHeap.Pop())
	}
	return h.selectHeuristicNeighborsFromMinHeap(minHeap, maxCnt)
}

func (h *HNSW) searchAtLayer(query []float32, enterPoint *data.Doc, ef, layer int32) *util.Heap {
	candidates, result := util.NewMinHeap(), util.NewMaxHeap()
	ele := &data.Element{
		Doc:      enterPoint,
		Distance: h.DisFunc(query, enterPoint.Vector),
	}
	candidates.Push(ele)
	result.Push(ele)
	visited := make(map[int32]struct{})
	visited[enterPoint.Id] = struct{}{}
	for candidates.Size() > 0 {
		candidate := candidates.Pop().(*data.Element)
		if candidate.Distance > result.Top().GetValue() {
			break
		}
		for _, n := range h.Neighbors[candidate.Doc.Id][layer] {
			if _, ok := visited[n.Doc.Id]; ok {
				continue
			}
			visited[n.Doc.Id] = struct{}{}
			newEle := &data.Element{
				Doc:      n.Doc,
				Distance: h.DisFunc(n.Doc.Vector, query),
			}
			h.ComputeCnt++
			if int32(result.Size()) < ef {
				result.Push(newEle)
				candidates.Push(newEle)
			} else if result.Top().GetValue() > newEle.Distance {
				result.PopAndPush(newEle)
				candidates.Push(newEle)
			}
		}
	}
	return result
}

func (h *HNSW) searchAtLayerWith1Ef(query []float32, enterPoint *data.Doc, layer int32) *data.Doc {
	maxDis := h.DisFunc(enterPoint.Vector, query)
	for {
		findBetter := false
		for _, n := range h.Neighbors[enterPoint.Id][layer] {
			dis := h.DisFunc(query, n.Doc.Vector)
			h.ComputeCnt++
			if dis < maxDis {
				enterPoint = n.Doc
				maxDis = dis
				findBetter = true
			}
		}
		if !findBetter {
			break
		}
	}
	return enterPoint
}

func (h *HNSW) addNeighbor(neighbors []*Neighbor, newNeighbor *Neighbor, layer int32) []*Neighbor {
	maxCnt := h.getMaxNeighborCnt(layer)
	neighbors = append(neighbors, newNeighbor)
	if h.Mode == Simple {
		// h.Neighbors[curDoc.Id][layer] is sorted
		idx := len(neighbors) - 1
		for i := len(neighbors) - 2; i >= 0; i-- {
			if newNeighbor.Dis >= neighbors[i].Dis {
				break
			}
			neighbors[i], neighbors[idx] = neighbors[idx], neighbors[i]
			idx = i
		}
		if int32(len(neighbors)) > maxCnt {
			neighbors = neighbors[0:maxCnt]
		}
		return neighbors
	} // else h.Mode == Heuristic
	minHeap := util.NewMinHeap()
	for _, neighbor := range neighbors {
		minHeap.Push(&data.Element{
			Doc:      neighbor.Doc,
			Distance: neighbor.Dis,
		})
	}
	return h.selectHeuristicNeighborsFromMinHeap(minHeap, maxCnt)
}

func (h *HNSW) SearchKNN(query []float32, ef, k, ignoreLayer int32) []*data.Doc {
	entryPoint := h.EntryPoint
	if ignoreLayer == 0 {
		for layer := h.MaxLayer; layer > 0; layer-- {
			entryPoint = h.searchAtLayerWith1Ef(query, entryPoint, layer)
		}
	}
	result := h.searchAtLayer(query, entryPoint, ef, 0)
	for result.Size() > int(k) {
		result.Pop()
	}
	list := make([]*data.Doc, result.Size())
	for i := result.Size() - 1; result.Size() > 0; i-- {
		list[i] = result.Pop().(*data.Element).Doc
	}
	return list
}

func (h *HNSW) getMaxNeighborCnt(layer int32) int32 {
	if layer == 0 {
		return h.M0
	}
	return h.M
}
