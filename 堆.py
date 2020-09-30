# -*- coding:utf-8 -*-

'''
堆练习
https://leetcode-cn.com/problems/top-k-frequent-elements/solution/python-dui-pai-xu-by-xxinjiee/
'''




# 215. 数组中的第K个最大元素
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        def down(start, end):
            if 2 * start + 1 <= end:
                maxChild = 2*start+2 if 2*start+2<=end and nums[2*start+2]>nums[2*start+1] else 2*start+1
                if nums[start] < nums[maxChild]:
                    nums[start], nums[maxChild] = nums[maxChild], nums[start]
                    down(maxChild, end)
        
        # 建最大堆
        for i in range(len(nums)//2-1, -1, -1):
            down(i, len(nums)-1)
        print '建堆后数组：', nums
        # 把堆顶也就是最大值删除并对剩下的元素重新堆排序
        for i in range(1, k+1):
            ret = nums[0]
            nums[0], nums[-i] = nums[-i], nums[0]
            down(0, len(nums)-1-i)            
            print '第{}次排序：'.format(i), nums

        return ret

s = Solution()
print s.findKthLargest([1,10,2,5,4,7,0], 6)
