package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/shiyinong/hnsw-go/algo/brute_force"
	"github.com/shiyinong/hnsw-go/algo/hnsw"
	"github.com/shiyinong/hnsw-go/algo/nsw"
	"github.com/shiyinong/hnsw-go/benchmark/hnsw_wrap"
	"github.com/shiyinong/hnsw-go/data"
	"github.com/shiyinong/hnsw-go/distance"
)

func buildHnsw() {
	docs := data.BuildAllDoc(int32(*dim), int32(*dataCount))
	hnswIdx := hnsw.BuildHNSW(int32(*hnswM), int32(*hnswEfCons), hnsw.Mode(*hnswMode), disType)
	start, s1 := time.Now(), time.Now()
	for i, doc := range docs {
		hnswIdx.Insert(doc)
		if (i+1)%10000 == 0 {
			fmt.Printf("HNSW index insert count: [%v], cost time: [%v]\n", i+1, time.Since(s1))
			s1 = time.Now()
		}
	}
	fmt.Printf("HNSW build index cost time: [%v]\n", time.Since(start))
	fmt.Printf("HNSW insertion avg compution cnt: [%v]\n", int(hnswIdx.ComputeCnt)/len(docs))

	nswIdx := nsw.BuildNSW(docs, int32(*nswF), int32(*nswW), disType)
	testDocs := data.BuildAllDoc(int32(*dim), int32(*testCount))
	bfRes := testBruteForce(docs, testDocs)
	wrap := &hnsw_wrap.HnswWrap{
		Hnsw:     hnswIdx,
		Nsw:      nswIdx,
		TestData: testDocs,
		TopK:     bfRes,
	}
	hnsw_wrap.SaveHnswWrap(wrap, *hnswFilaPath)
}

func testHnsw() {
	wrap := hnsw_wrap.LoadHnswWrap(*hnswFilaPath)
	wrap.Hnsw.Ef = int32(*hnswEf)
	hnswRes, nswRes := [][]*data.Doc{}, [][]*data.Doc{}
	start := time.Now()
	for _, doc := range wrap.TestData {
		knn := wrap.Nsw.SearchKNN(doc.Vector, int32(*k), int32(*nswM))
		nswRes = append(nswRes, knn)
	}
	cost := time.Since(start).Milliseconds()
	fmt.Println("--------------- NSW ------------------")
	compare(wrap.TopK, nswRes)
	fmt.Printf("NSW query performance: [%.2f ms / query], ", float64(cost)/float64(len(wrap.TestData)))
	fmt.Printf("[%0.f queries / second]\n", 1000*float64(len(wrap.TestData))/float64(cost))
	fmt.Printf("NSW query avg compution cnt: [%v]\n", int(wrap.Nsw.ComputeCnt)/len(wrap.TestData))
	wrap.Nsw.Stat()

	start = time.Now()
	for _, doc := range wrap.TestData {
		knn := wrap.Hnsw.SearchKNN(doc.Vector, int32(*hnswEf), int32(*k), int32(*hnswIgnoreLayer))
		hnswRes = append(hnswRes, knn)
	}
	cost = time.Since(start).Milliseconds()
	fmt.Println("--------------- HNSW ------------------")
	compare(wrap.TopK, hnswRes)
	fmt.Printf("HNSW query performance: [%.2f ms / query], ", float64(cost)/float64(len(wrap.TestData)))
	fmt.Printf("[%0.f queries / second]\n", 1000*float64(len(wrap.TestData))/float64(cost))
	fmt.Printf("HNSW query avg compution cnt: [%v]\n", int(wrap.Hnsw.ComputeCnt)/len(wrap.TestData))
	wrap.Hnsw.Stat()
}

func testBruteForce(docs, testDocs []*data.Doc) [][]*data.Doc {
	start := time.Now()
	bf := &brute_force.Searcher{Docs: docs}
	res := [][]*data.Doc{}
	for i := 0; i < len(testDocs); i++ {
		knn := bf.Query(testDocs[i].Vector, int32(*k), disType)
		res = append(res, knn)
	}
	cost := time.Since(start).Milliseconds()
	fmt.Printf("Brute Force query performance: [%.2f ms / query], ", float64(cost)/float64(len(testDocs)))
	fmt.Printf("[%0.f queries / second]\n", 1000*float64(len(testDocs))/float64(cost))
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
	fmt.Printf("recall rate: [%.4f%%]\n", 100*float64(hitCount)/float64(allCount))
}

const (
	disType = distance.L2
)

var (
	dim       = flag.Int("d", 8, "")
	k         = flag.Int("k", 10, "")
	dataCount = flag.Int("count", 100000, "")
	testCount = flag.Int("test_count", 1000, "")
	operation = flag.String("operation", "", "")

	nswF = flag.Int("nsw_f", 20, "")
	nswW = flag.Int("nsw_w", 2, "")
	nswM = flag.Int("nsw_m", 2, "")

	hnswM           = flag.Int("hnsw_m", 6, "")
	hnswEf          = flag.Int("hnsw_ef", 64, "")
	hnswEfCons      = flag.Int("hnsw_ef_cons", 32, "")
	hnswMode        = flag.Int("hnsw_mode", 1, "")
	hnswFilaPath    = flag.String("hnsw_file_path", "./hnsw_8d.data", "")
	hnswIgnoreLayer = flag.Int("hnsw_ignore_layer", 0, "")
)

func main() {
	flag.Parse()

	if *operation == "only_build" {
		buildHnsw()
	} else if *operation == "only_test" {
		testHnsw()
	} else {
		buildHnsw()
		testHnsw()
	}
}
