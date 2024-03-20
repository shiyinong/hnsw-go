package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/shiyinong/hnsw-go/algo/brute_force"
	"github.com/shiyinong/hnsw-go/algo/hnsw"
	"github.com/shiyinong/hnsw-go/algo/nsw"
	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
)

func buildHnsw(docs []*data.Doc) *hnsw.HNSW {
	hnswIdx := hnsw.BuildHNSW(*hnswM, *hnswEfCons, 0, disType)
	start := time.Now()
	for i, doc := range docs {
		hnswIdx.Insert(doc.Vector)
		if (i+1)%10000 == 0 {
			fmt.Printf("HNSW index insert count: [%v], cost time: [%v]\n", i+1, time.Since(start))
			start = time.Now()
		}
	}
	return hnswIdx
}

func testHnsw(docs []*data.Doc, hnswIdx *hnsw.HNSW) [][]*data.Doc {
	res := [][]*data.Doc{}
	for i := 0; i < len(docs); i++ {
		knn := hnswIdx.SearchKNN(docs[i].Vector, *hnswEf, *k)
		res = append(res, knn)
	}
	return res
}

func testNsw(docs []*data.Doc, nswIdx *nsw.NSW) [][]*data.Doc {
	res := [][]*data.Doc{}
	for i := 0; i < len(docs); i++ {
		knn := nswIdx.SearchKNN(docs[i].Vector, *k)
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
	disType = distance.L2
)

var (
	dim       = flag.Int("d", 16, "")
	k         = flag.Int("k", 10, "")
	dataCount = flag.Int("count", 100000, "")
	testCount = flag.Int("test_count", 1000, "")

	hnswM      = flag.Int("hnsw_m", 6, "")
	hnswEf     = flag.Int("hnsw_ef", 64, "")
	hnswEfCons = flag.Int("hnsw_ef_cons", 64, "")
)

func main() {
	flag.Parse()
	dataDocs := data.BuildAllDoc(int32(*dim), int32(*dataCount))
	testDocs := data.BuildAllDoc(int32(*dim), int32(*testCount))

	bf := &brute_force.Searcher{Docs: dataDocs}
	start := time.Now()

	bfRes := testBruteForce(testDocs, bf)
	fmt.Printf("BF query cost: [%v]\n", time.Since(start))
	//
	//start = time.Now()
	//nswIdx := nsw.BuildNSW(dataDocs, (*dim)*2, disType)
	//fmt.Println("\n-------------- NSW --------------")
	//fmt.Printf("build NSW index cost: [%v]\n", time.Since(start))
	//
	//start = time.Now()
	//nswRes := testNsw(testDocs, nswIdx)
	//fmt.Printf("NSW query cost: [%v]\n", time.Since(start))
	//compare(bfRes, nswRes)

	start = time.Now()
	hnswIdx := buildHnsw(dataDocs)
	fmt.Println("\n-------------- HNSW --------------")
	fmt.Printf("build HNSW index cost: [%v]\n", time.Since(start))

	start = time.Now()
	hnswRes := testHnsw(testDocs, hnswIdx)
	fmt.Printf("HNSW query cost: [%v]\n", time.Since(start))
	compare(bfRes, hnswRes)
	hnswIdx.Stat()
}
