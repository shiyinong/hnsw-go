package nsw

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
	"github.com/shiyinong/hnsw-go/util"
)

type NSW struct {
	docs []*data.Doc
	// id of doc
	links [][]int32
	// count of node neighbors
	f       int
	disFunc func(vec1, vec2 []float32) float32
}

func BuildNSW(docs []*data.Doc, f int, disType distance.Type) *NSW {
	if len(docs) == 0 {
		panic("data is nil")
	}
	docCount := len(docs)
	nsw := &NSW{
		docs:    make([]*data.Doc, 0, docCount),
		links:   make([][]int32, docCount),
		f:       f,
		disFunc: distance.FuncMap[disType],
	}
	start := time.Now()
	for _, curDoc := range docs {
		neighbors := nsw.docs
		if len(nsw.docs) > nsw.f {
			neighbors = nsw.SearchKNN(curDoc.Vector, nsw.f)
		}
		nsw.docs = append(nsw.docs, curDoc)
		for _, neighbor := range neighbors {
			nsw.links[neighbor.Id] = append(nsw.links[neighbor.Id], curDoc.Id)
			nsw.links[curDoc.Id] = append(nsw.links[curDoc.Id], neighbor.Id)
		}
		if len(nsw.docs)%10000 == 0 {
			fmt.Printf("NSW index insert count: [%v], cost time: [%v]\n", len(nsw.docs), time.Since(start))
			start = time.Now()
		}
	}
	return nsw
}

func (n *NSW) SearchKNN(query []float32, k int) []*data.Doc {
	/*
		1. build a min heap named candidates, build a max heap(size: k) named results.
		2. get an entry Node by random, put it to the candidates and results.
		3. pop a node(named C) from candidates top, if C is further than results top, then end.
		4. else foreach every neighbor of C, add to candidates, results.
	*/
	visited := map[int32]struct{}{}
	results, candidates := util.NewMaxHeap(), util.NewMinHeap()
	entry := n.docs[rand.Int31n(int32(len(n.docs)))]
	entryEle := &data.Element{
		Doc:      entry,
		Distance: n.disFunc(entry.Vector, query),
	}
	candidates.Push(entryEle)
	for candidates.Size() > 0 {
		cur := candidates.Pop().(*data.Element)
		if results.Size() > 0 && cur.Distance > results.Top().GetValue() {
			break
		}
		for _, neighborIdx := range n.links[cur.Doc.Id] {
			neighbor := n.docs[neighborIdx]
			if _, ok := visited[neighbor.Id]; ok {
				continue
			}
			visited[neighbor.Id] = struct{}{}
			ele := &data.Element{
				Doc:      neighbor,
				Distance: n.disFunc(neighbor.Vector, query),
			}
			if results.Size() < k {
				results.Push(ele)
				candidates.Push(ele)
			} else if results.Top().GetValue() > ele.Distance {
				results.PopAndPush(ele)
				candidates.Push(ele)
			}
		}
	}
	topK := make([]*data.Doc, k)
	for i := len(topK) - 1; i >= 0; i-- {
		doc := results.Pop().(*data.Element).Doc
		topK[i] = doc
	}
	return topK
}
