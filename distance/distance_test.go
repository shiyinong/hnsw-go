package distance

import (
	"fmt"
	"testing"

	"github.com/shiyinong/hnsw-go/data"
)

func TestL2Closeness(t *testing.T) {
	v1 := data.BuildDoc(1, 1)
	v2 := data.BuildDoc(1, 1)

	dis := L2Distance(v1.Vector, v2.Vector)
	fmt.Println(dis)
}
