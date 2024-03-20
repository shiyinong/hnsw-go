package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/shiyinong/hnsw-go/algo/brute_force"
	"github.com/shiyinong/hnsw-go/algo/nsw"
	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
)

func testNsw(docs []*data.Doc, nswIdx *nsw.NSW) [][]*data.Doc {
	res := [][]*data.Doc{}
	for i := 0; i < len(docs); i++ {
		knn := nswIdx.SearchKNN(docs[i].Vector, *k, nswM)
		res = append(res, knn)
	}
	return res
}

func testBruteForce(docs []*data.Doc, bf *brute_force.Searcher) [][]*data.Doc {
	res := [][]*data.Doc{}
	for i := 0; i < len(docs); i++ {
		knn := bf.Query(docs[i].Vector, int32(*k), disType)
		res = append(res, knn)
	}
	return res
}

func compare(res1, res2 [][]*data.Doc) {
	hitCount, allCount := 0, 0
	for i, docs := range res1 {
		m := make(map[int32]struct{})
		for _, doc := range docs {
			allCount++
			m[doc.Id] = struct{}{}
		}
		for _, doc := range res2[i] {
			if _, ok := m[doc.Id]; ok {
				hitCount++
			}
		}
	}
	fmt.Printf("recall rate: [%.2f%%]\n", 100*float64(hitCount)/float64(allCount))
}

const (
	nswW    = 1
	nswM    = 1
	disType = distance.L2
)

var (
	dim       = flag.Int("d", 16, "")
	k         = flag.Int("k", 100, "")
	dataCount = flag.Int("count", 200000, "")
	testCount = flag.Int("test_count", 1000, "")
)

func main() {
	flag.Parse()
	newF := (*dim) * 2
	dataDocs := data.BuildAllDoc(int32(*dim), int32(*dataCount))
	testDocs := data.BuildAllDoc(int32(*dim), int32(*testCount))

	bf := &brute_force.Searcher{Docs: dataDocs}
	start := time.Now()
	nswIdx := nsw.BuildNSW(dataDocs, nswW, newF, disType)
	fmt.Printf("build NSW index cost: [%v]\n", time.Since(start))

	for i := 0; i < 1; i++ {
		start = time.Now()
		bfRes := testBruteForce(testDocs, bf)
		fmt.Printf("BF query cost: [%v]\n", time.Since(start))

		start = time.Now()
		nswRes := testNsw(testDocs, nswIdx)
		fmt.Printf("NSW query cost: [%v]\n", time.Since(start))

		compare(bfRes, nswRes)
	}
}
