class Solution (object):
    def calculator(self, num1, num2, operator) -> float:
        if (str(operator)) == '+':
            output = num1 + num2

        elif (str(operator)) == "-":
            output = num1 - num2

        elif (str(operator)) == "*":
            output = num1 * num2

        else:
            if num2 == 0:
                print ("Cannot divide by zero")
                return None
            else:
                output = num1 / num2

        return output

s = Solution()
output = s.calculator(5,2,'+')
print (f"The final output is: {output}")




