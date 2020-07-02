class Solution(object):
    def partitionLabels(self, S):

        last = {c:i for i,c in enumerate(S)}
        
        start = 0; end = last[S[0]]; cnt = 0; r = []
        for i,c in enumerate(S):
            end = max(end, last[c])
            if i == end:
                r.append(end-start+1)
                if i < len(S)-1:
                    start = i+1
                    end = last[S[i+1]]
        return r
            