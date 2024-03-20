package util

func Min[T int32 | int64 | int](n1, n2 T) T {
	if n1 < n2 {
		return n1
	}
	return n2
}

func Max[T int32 | int64 | int](n1, n2 T) T {
	if n1 > n2 {
		return n1
	}
	return n2
}
