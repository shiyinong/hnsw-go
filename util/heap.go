package util

type HeapEle interface {
	GetValue() float32
}

type Heap struct {
	Elements []HeapEle
	less     func(e1, e2 HeapEle) bool
}

func NewHeap(f func(e1, e2 HeapEle) bool) *Heap {
	return &Heap{
		less: f,
	}
}

func NewMinHeap() *Heap {
	return NewHeap(
		func(e1, e2 HeapEle) bool {
			return e1.GetValue() <= e2.GetValue()
		},
	)
}

func NewMaxHeap() *Heap {
	return NewHeap(
		func(e1, e2 HeapEle) bool {
			return e1.GetValue() >= e2.GetValue()
		},
	)
}

func (h *Heap) Size() int {
	return len(h.Elements)
}

func (h *Heap) Top() HeapEle {
	if h.Size() == 0 {
		return nil
	}
	return h.Elements[0]
}

func (h *Heap) Pop() HeapEle {
	if h.Size() == 0 {
		return nil
	}
	res := h.Elements[0]
	h.Elements[0] = h.Elements[h.Size()-1]
	h.Elements = h.Elements[0 : h.Size()-1]
	h.fixDown(0)
	return res
}

func (h *Heap) Push(ele HeapEle) {
	h.Elements = append(h.Elements, ele)
	h.fixUp(h.Size() - 1)
}

func (h *Heap) PopAndPush(ele HeapEle) HeapEle {
	if h.Size() == 0 {
		return nil
	}
	res := h.Elements[0]
	h.Elements[0] = ele
	h.fixDown(0)
	return res
}

func (h *Heap) fixUp(child int) {
	for {
		parent := (child - 1) / 2
		if parent < 0 || h.less(h.Elements[parent], h.Elements[child]) {
			return
		}
		h.Elements[parent], h.Elements[child] = h.Elements[child], h.Elements[parent]
		child = parent
	}
}

func (h *Heap) fixDown(parent int) {
	for {
		minChild := 2*parent + 1 // left
		if minChild >= h.Size() {
			break
		}
		if minChild+1 < h.Size() && h.less(h.Elements[minChild+1], h.Elements[minChild]) {
			minChild++ // right
		}
		if !h.less(h.Elements[minChild], h.Elements[parent]) {
			return
		}
		h.Elements[minChild], h.Elements[parent] = h.Elements[parent], h.Elements[minChild]
		parent = minChild
	}
}
