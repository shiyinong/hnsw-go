package brute_force

import (
	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
	"github.com/shiyinong/hnsw-go/util"
)

type Searcher struct {
	Docs []*data.Doc
}

func (s *Searcher) Query(query []float32, k int32, disType distance.Type) []*data.Doc {
	topK := util.NewMaxHeap()

	lengthMap := make(map[int]int)
	disFunc := distance.FuncMap[disType]
	for _, doc := range s.Docs {
		ele := &data.Element{
			Doc:      doc,
			Distance: disFunc(doc.Vector, query),
		}
		lengthMap[int(1000*ele.Distance)]++
		if topK.Size() < int(k) {
			topK.Push(ele)
			continue
		}

		dis := disFunc(query, topK.Top().(*data.Element).Doc.Vector)
		if ele.Distance >= dis {
			continue
		}
		topK.PopAndPush(ele)
	}

	res := []*data.Doc{}
	for topK.Size() > 0 {
		res = append(res, topK.Pop().(*data.Element).Doc)
	}
	return res
}
