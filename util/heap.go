package util

type HeapEle interface {
	GetValue() float32
}

type Heap struct {
	elements []HeapEle
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
	return len(h.elements)
}

func (h *Heap) Top() HeapEle {
	if h.Size() == 0 {
		return nil
	}
	return h.elements[0]
}

func (h *Heap) Pop() HeapEle {
	if h.Size() == 0 {
		return nil
	}
	res := h.elements[0]
	h.elements[0] = h.elements[h.Size()-1]
	h.elements = h.elements[0 : h.Size()-1]
	h.fixDown(0)
	return res
}

func (h *Heap) Push(ele HeapEle) {
	h.elements = append(h.elements, ele)
	h.fixUp(h.Size() - 1)
}

func (h *Heap) PopAndPush(ele HeapEle) HeapEle {
	if h.Size() == 0 {
		return nil
	}
	res := h.elements[0]
	h.elements[0] = ele
	h.fixDown(0)
	return res
}

func (h *Heap) fixUp(child int) {
	for {
		parent := (child - 1) / 2
		if parent < 0 || h.less(h.elements[parent], h.elements[child]) {
			return
		}
		h.elements[parent], h.elements[child] = h.elements[child], h.elements[parent]
		child = parent
	}
}

func (h *Heap) fixDown(parent int) {
	for {
		minChild := 2*parent + 1 // left
		if minChild >= h.Size() {
			break
		}
		if minChild+1 < h.Size() && h.less(h.elements[minChild+1], h.elements[minChild]) {
			minChild++ // right
		}
		if !h.less(h.elements[minChild], h.elements[parent]) {
			return
		}
		h.elements[minChild], h.elements[parent] = h.elements[parent], h.elements[minChild]
		parent = minChild
	}
}
