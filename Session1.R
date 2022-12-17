print("Hello")

print(mtcars)

veriaty1 <- order(-mtcars$mpg)
mtcars[veriaty1,]


veriaty2 <- mtcars[mtcars$cyl >= 6,]
print(veriaty2)


variaty3 <- mean(mtcars$mpg)
print(variaty3)

variaty4 <- mean(mtcars$cyl)
print(variaty4)

variaty5 <- mean(mtcars$disp)
print(variaty5)

variaty6 <- mean(mtcars$hp)
print(variaty6)

variaty7 <- mean(mtcars$drat)
print(variaty7)

variaty8 <- mean(mtcars$wt)
print(variaty8)

variaty9 <- mean(mtcars$qsec)
print(variaty9)

variaty10 <- mean(mtcars$gear)
print(variaty10)

variaty11 <- mean(mtcars$carb)
print(variaty11)

means_of_col <- colMeans(mtcars)
print(means_of_col)




# Q2
question2 = sd(mtcars$mpg)
print(question2)
print(question2*10)



# Q3
question3 <- toupper('aaaaaaaaaaaa')  
print(question3)



















