

# 两个队列实现栈
# https://blog.csdn.net/su_bao/article/details/82347567?utm_source=blogkpcl3

"""
push()操作：
为了保证先进栈的元素一直在栈底，需要将两个队列交替使用，才能满足需求。
因此，想法是，我们只在空的那个队列上添加元素，然后把非空的那个队列中的元素全部追加到当前这个队列。这样一来，我们又得到一个空的队列，供下一次添加元素。

pop()操作：
因为在添加元素时，我们已经按照进栈的先后顺序把后进栈的元素放在一个队列的头部，所以出栈操作时，我们只需要找到那个非空的队列，并依次取出数据即可。
"""



from collections import deque

class StackWithTwoQueues(object):
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
 
    def push(self,x):
        if len(self.q1) == 0: self.q1.append(x)
        elif len(self.q2) == 0: self.q2.append(x)
        if len(self.q2) == 1 and len(self.q1) >= 1: while self.q1: self.q2.append(self.q1.popleft()) # ">=1"中的等号指刚开始两个队列各有一个元素的时候
        elif len(self.q1) == 1 and len(self.q2) > 1: while self.q2: self.q1.append(self.q2.popleft())
    
    def pop(self):
        if self.q1: return self.q1.popleft()
        elif self.q2: return self.q2.popleft()
        return None


# 每日温度
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        
        s = []
        ret = [0 for _ in range(len(T))]
        for i, t in enumerate(reversed(T)):
            I = len(T) - 1 - i
            while s and t >= s[-1][1]:
                s.pop()
            ret[I] = (s[-1][0] - I) if s else 0
            s.append((I, t))
        return ret

# 下一个更大的元素
class Solution:
    def nextGreaterElement(self, nums1, nums2):
        d = {}
        s = []
        for n in reversed(nums2):
            while s and n >= s[-1]: s.pop()
            bigger = s[-1] if s else -1
            d[n] = bigger
            s.append(n)
        
        ret = []
        for n in nums1:
            ret.append(d[n])
        return ret            
    

# 503. 下一个更大元素 II   
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        ret = []
        s = reversed(nums[:-1])
        for n in reversed(nums):
            while s and n >= s[-1]:
                s.pop()
            ret.insert(0, s[-1] if s else -1)
        return ret
            
    