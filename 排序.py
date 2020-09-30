
# # 快排
# def quick_sort(l):
#     if len(l) <= 1: return l
#     mid_value = l[len(l) // 2]
#     left = [x for x in l if x < mid_value]
#     right = [x for x in l if x > mid_value]
#     middle = [x for x in l if x == mid_value]
#     return quick_sort(left) + middle + quick_sort(right)


# # 合并俩数组
# def mergeTwoArray(l1, l2):
#     if not l1: return l2
#     if not l2: return l1
    
#     l = []
#     i=j=0
#     while i<len(l1) and j<len(l2):
#         if l1[i] < l2[j]:
#             l.append(l1[i])
#             i+=1
#         else:
#             l.append(l2[j])
#             j+=1
    
#     if i<len(l1):
#         l.extend(l1[i:])
#     if j<len(l2):
#         l.extend(l2[j:])
    
#     return l


class Solution:
    def majorityElement(self, numbers):
        # write code here
        
        votes = 0
        for num in numbers:
            if votes == 0:
                x = num
            if num == x:
                votes += 1
            else:
                votes -= 1
        return x
    
    
class Solution:
    def majorityElement(self, nums):
        votes = 0
        for num in nums:
            if votes == 0: x = num
            votes += 1 if num == x else -1
        return x




s = Solution()
print(s.majorityElement([1,2,3,2,4,2,5,2,3]))

    