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
	Docs []*data.Doc
	// id of doc
	Links [][]int32
	// count of node neighbors
	F          int32
	W          int32
	DisFunc    func(vec1, vec2 []float32) float32
	ComputeCnt int64
}

func (n *NSW) Stat() {
	cnt := 0
	for _, link := range n.Links {
		cnt += len(link)
	}
	fmt.Printf("NSW node avg neighbors count: [%v]\n", cnt/len(n.Docs))
}

func BuildNSW(docs []*data.Doc, f, w int32, disType distance.Type) *NSW {
	if len(docs) == 0 {
		panic("data is nil")
	}
	docCount := len(docs)
	nsw := &NSW{
		Docs:    make([]*data.Doc, 0, docCount),
		Links:   make([][]int32, docCount),
		F:       f,
		W:       w,
		DisFunc: distance.FuncMap[disType],
	}
	start, s1 := time.Now(), time.Now()
	for _, curDoc := range docs {
		neighbors := nsw.Docs
		if len(nsw.Docs) > int(nsw.F) {
			neighbors = nsw.SearchKNN(curDoc.Vector, nsw.F, nsw.W)
		}
		nsw.Docs = append(nsw.Docs, curDoc)
		for _, neighbor := range neighbors {
			nsw.Links[neighbor.Id] = append(nsw.Links[neighbor.Id], curDoc.Id)
			nsw.Links[curDoc.Id] = append(nsw.Links[curDoc.Id], neighbor.Id)
		}
		if len(nsw.Docs)%10000 == 0 {
			fmt.Printf("NSW index insert count: [%v], cost time: [%v]\n", len(nsw.Docs), time.Since(s1))
			s1 = time.Now()
		}
	}
	fmt.Printf("NSW build index cost time: [%v]\n", time.Since(start))
	fmt.Printf("NSW insertion avg compution cnt: [%v]\n", int(nsw.ComputeCnt)/len(docs))
	return nsw
}

func (n *NSW) SearchKNN(query []float32, k, m int32) []*data.Doc {
	/*
		1. build a min heap named candidates, build a max heap(size: k) named results.
		2. get an entry Node by random, put it to the candidates and results.
		3. pop a node(named C) from candidates top, if C is further than results top, then end.
		4. else foreach every neighbor of C, add to candidates, results.
	*/
	visited := map[int32]struct{}{}
	results, candidates := util.NewMaxHeap(), util.NewMinHeap()
	for i := int32(0); i < m; i++ {
		entry := n.Docs[rand.Int31n(int32(len(n.Docs)))]
		entryEle := &data.Element{
			Doc:      entry,
			Distance: n.DisFunc(entry.Vector, query),
		}
		candidates.Push(entryEle)
		visited[entry.Id] = struct{}{}
		for candidates.Size() > 0 {
			cur := candidates.Pop().(*data.Element)
			if results.Size() > 0 && cur.Distance > results.Top().GetValue() {
				break
			}
			for _, neighborIdx := range n.Links[cur.Doc.Id] {
				neighbor := n.Docs[neighborIdx]
				if _, ok := visited[neighbor.Id]; ok {
					continue
				}
				n.ComputeCnt++
				visited[neighbor.Id] = struct{}{}
				ele := &data.Element{
					Doc:      neighbor,
					Distance: n.DisFunc(neighbor.Vector, query),
				}
				candidates.Push(ele)
				if results.Size() < int(k) {
					results.Push(ele)
				} else if results.Top().GetValue() > ele.Distance {
					results.PopAndPush(ele)
				}
			}
		}
	}
	topK := make([]*data.Doc, results.Size())
	for i := len(topK) - 1; i >= 0; i-- {
		doc := results.Pop().(*data.Element).Doc
		topK[i] = doc
	}
	return topK
}
