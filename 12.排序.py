
# # 快排 (并归)
def quick_sort(a):
    if len(a) <= 1: return a
    m = len(a) // 2
    left = [x for x in a if x < m]
    mid = [x for x in a if x == m]
    right = [x for x in a if x > m]
    return quick_sort(left) + mid + quick_sort(right)


## 合并俩【有序】数组 (双指针)
def mergeTwoArray(l1, l2):
    if not l1: return l2
    if not l2: return l1
    
    i = j = 0
    ans = []
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]: ans.append(l1[i]); i += 1
        else: ans.append(l2[j]); j += 1
    if i < len(l1): ans.extend(l1[i:])
    if j < len(l2): ans.extend(l2[i:])
    return ans
    
    
# 合并多个有序数组 (并归)
def mergeLists(lists):
    if not lists: return []
    if len(lists) == 1: return lists[0]
    mid = len(lists) // 2
    left = mergeLists(lists[:mid])
    right = mergeLists(lists[mid:])
    return mergeTwoArray(left, right)


# 435. 无重叠区间
class Solution:
    def eraseOverlapIntervals(self, A: List[List[int]]) -> int:
        count = 0
        A.sort(key=lambda x: x[1]) # 按照结束时间排序，若按开始时间，需要倒遍历
        pre = A[0]
        for x in A[1:]:
            if x[0] < pre[1]: # 若果交叉，count++,代表需要删除数量
                count += 1
            else:
                pre = x
        return count