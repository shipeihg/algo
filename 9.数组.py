

# 查找数组中大于一半数量的元素
class Solution:
    def majorityElement(self, numbers):
        # write code here
        votes = 0
        for num in numbers:
            if votes == 0: x = num
            if num == x: votes += 1
            else: votes -= 1
        return x

        
# 485. 最大连续1的个数
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxOnes = curOnes = 0
        for n in nums:
            if n == 0: curOnes = 0
            else: curOnes += 1
            maxOnes = max(maxOnes, curOnes)
        return maxOnes
    
                
# 240. 搜索二维矩阵 II
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        
        # if not matrix or not matrix[0]: return False
        # m, n = len(matrix), len(matrix[0])
        # i, j = 0, n-1
        # while i < m and j >= 0:
        #     if matrix[i][j] == target: return True
        #     elif matrix[i][j] > target: j -= 1
        #     else: i += 1
        # return False

        if not matrix or not matrix[0]: return False
        m, n = len(matrix), len(matrix[0])
        i, j = m-1, 0
        while i >= 0 and j < n:
            if matrix[i][j] == target: return True
            elif matrix[i][j] > target: i -= 1
            else: j += 1
        return False


# 59. 螺旋矩阵II
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        
        left, right, up, down = 0, n-1, 0, n-1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        a = 0
        while a < n*n:
            for i in range(left, right+1, 1): a += 1; mat[up][i] = a
            up += 1
            for i in range(up, down+1, 1): a += 1; mat[i][right] = a
            right -= 1
            for i in range(right, left-1, -1): a += 1; mat[down][i] = a
            down -= 1
            for i in range(down, up-1, -1): a += 1; mat[i][left] = a
            left += 1
        return mat

# 54. 螺旋矩阵
#https://leetcode-cn.com/problems/spiral-matrix/solution/shou-hui-tu-jie-liang-chong-bian-li-de-ce-lue-kan-/
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        if not matrix: return []
        if not matrix[0]: [[] for _ in range(len(matrix))]
        
        m, n = len(matrix), len(matrix[0])
        left, right, up, down = 0, n-1, 0, m-1
        cnt = 0
        ret = []
        while cnt < m*n:
            for i in range(left, right+1, 1): ret.append(matrix[up][i]); cnt +=1
            up += 1
            for i in range(up, down+1, 1): ret.append(matrix[i][right]); cnt += 1
            right -= 1
            for i in range(right, left-1, -1): ret.append(matrix[down][i]); cnt += 1
            down -= 1
            for i in range(down, up-1, -1): ret.append(matrix[i][left]); cnt += 1
            left += 1
        return ret


# 766. 托普利茨矩阵         
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        m, n = len(matrix), len(matrix[0])
        return all(matrix[i][j]==matrix[i-1][j-1] for i in range(1,m) for j in range(1,n))
  
                
        
        