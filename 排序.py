
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