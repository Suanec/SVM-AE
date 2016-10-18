/**
 * 单调函数方程求解。
 *
 */

def fun( x : Double ) : Double = x*x*x // 7*x + 11
def fun( x : Double ) : Double = math.exp(x)
def abs( x : Double ) : Double = if( x >= 0 ) x else -x
val MINDOUBLE = 0.0000000001d




def calc( t : Double ) : Double = {
  var y = t
  var y1 = t
  var lastDown = true
  var i = t
  var x = t
  var x1 = t
  var yita = 25d
  var times = 1
  while( i > MINDOUBLE && times < 10000 ) {//0000000000000
    //if(times%10000 == 0) 
    println(s" x = $x, i = $i, y = $y, yita = $yita, times = $times, lastDown = $lastDown")
    y = fun(x)
    // if(lastDown){
      if( y > t ) x -= abs(yita)
      else if (y < t) x += abs(yita)//// 0.3
    // }else{
    //   if( y > t ) x += abs(yita)
    //   else if (y < t) x += abs(yita)//// 0.3
    // }
    if((y / y1 > 0)&&(abs(y-t)<abs(y1-t))) lastDown = true
    else lastDown = false
    i = abs(x - x1)
    x1 = x
    if(times < 1000){
    if(yita != 0 && yita > MINDOUBLE && yita < 100) yita = yita / (times/10+1) * 7 + 0.0001d 
    else yita = 1d
    }
    else {
      if(yita != 0 && yita > MINDOUBLE  && yita < 100) yita = yita / (times/100+1) * 7 + MINDOUBLE 
      else yita = 1d
    }
    times += 1
  }
  (x / 0.00000001 ).toLong * 0.00000001
}



calc(101)
