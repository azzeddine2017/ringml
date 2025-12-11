load "../src/ringml.ring"

t1 = new Tensor(1,1) { aData=[[10.0]] }
t2 = new Tensor(1,1) { aData=[[3.0]] }

# العملية المطلوبة: 10 - 3 = 7
t1.sub(t2)

see "Result of 10 - 3: "
t1.print()

if t1.aData[1][1] = 7
    see "SUB is CORRECT (A - B)" + nl
else
    see "SUB is REVERSED or WRONG!" + nl
ok