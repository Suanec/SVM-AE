-----------------------------------------------------------------------------
spark-shell --driver-memory 18G
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

type Svector = Array[Double]
type Sword = (Int,Int,Double)
type SwordBuf = ArrayBuffer[Sword]
type Sdoc = Array[Sword]
type Sdocs = ArrayBuffer[Sdoc]
case class dataStats( rows : Int, cols : Int, values : Sdocs) extends Serializable
case class NNConfig(
  size: Array[Int] = Array(10,2,10),
  layer: Int = 3,
  learningRate: Double = 0.1f,
  iteration : Int = 10,
  activation_function: String = "sigmoid",
  momentum: Double = 0.5f,
  scaling_learningRate: Double = 0.5f,
  weightPenaltyL2: Double = 0.5f,
  nonSparsityPenalty: Double = 0.5f,
  sparsityTarget: Double = 0.2f,
  inputZeroMaskedFraction: Double = 0.8f,
  dropoutFraction: Double = 0.3f,
  testing: Double = 0.1f,
  output_function: String = "sigmoid") extends Serializable



def Sigmoid( row : Svector ) : Svector = {
  row.map{
    x => (1 / (1 + Math.exp(-1 * x))).toDouble
  }
}

def softmax(arr : SwordBuf) = {
  var sum = 0.0
  var empty = true
  val res = new SwordBuf
  arr.foreach{
     i =>
     if(i._3 <= 709) sum += Math.exp(i._3).toDouble
     else empty = false
  }
  if(empty)  (0 until arr.size).map( i => (arr(i)._1, arr(i)._2, (Math.exp(arr(i)._3) /sum).toDouble) ).toArray
  else res.toArray
}


def readMTrick( file : String, minPartitions : Int = 5 ) : dataStats = {
  val data = sc.textFile( file, minPartitions ).collect
  val wordBuffer = new SwordBuf
  val docBuffer = new Sdocs
  var pre_rowNum = 1 
  var max_colNum = 1 
  
  val docs_data = data.map{
    line =>
      val splits = line.split(',')
      val m_word = new Sword(splits.head.toInt,splits(1).toInt,splits.last.toDouble)
      if( max_colNum < m_word._1 ) max_colNum = m_word._1
      if( m_word._2 == pre_rowNum ){
        wordBuffer += m_word
      }
      else{
        val doc = softmax(wordBuffer)
        if(doc.size != 0) docBuffer += doc.sortWith(_._1 < _._1) 
        wordBuffer.clear
        wordBuffer += m_word
        pre_rowNum = m_word._2
      }
      
      null
  }
  val doc = softmax(wordBuffer)
  if( doc.size != 0 ) docBuffer += doc.sortWith(_._1 < _._1) 
  new dataStats(docBuffer.size, max_colNum, docBuffer.sortWith(_.head._2 < _.head._2))
}/// readMTricks return dataStats


class Smatrix(
    private var rows : Int, 
    private var cols : Int, 
    private var values : Array[Svector], 
    private var rand : Random
    ) extends Serializable {
  def this() = this(0,0,null,new Random(134L))
  
  def this( values : Array[Svector] ){
    this( values.size.toInt, 0, values, null)
    this.rows = values.size.toInt
    values.map{
      i =>
        if( cols < i.size.toInt ) cols = i.size.toInt
    }
    this.values = new Array[Svector](rows)      
    (0 until values.size).map{
      i => 
        this.values(i) = new Svector(cols)
        values(i).copyToArray(this.values(i),0)
    }
  }///  def Smatrix( values : Array[Svector] )

  def this( rows : Int, cols : Int ){
    this( rows, cols, null, new Random(System.currentTimeMillis()) )
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
    } 
  }/// Smatrix zero
  
  def this( rows : Int, cols : Int, seed : Long ) = {
    this( rows, cols, null, new Random(seed))
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
        (0 until this.values(i).size).map{
          j => this.values(i)(j) = (rand.nextDouble()%0.8f) + 0.1f - 0.3f /// -0.2 ~ 0.6
        }
    }
  }/// Smatrix random

  def mulRow( row : Svector ) : Svector = {
    require( row.size == this.cols , "Dimision fault! --From Smatrix.mulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij
    }
    res
  }/// mulRow
  
  def netMulRow( row : Svector ) : Svector = {
    require( (row.size + 1) == this.cols , "Dimision fault! --From Smatrix.netMulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols -1 ).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// netMulRow
  
  def mulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols >= doc.last._1, "Dimision fault! --From Smatrix.mulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij
    }
    res
  }/// mulDoc
  
  def netMulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols+1 >= doc.last._1 + 1, "Dimision fault! --From Smatrix.netMulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// mulDoc
  
  def size() : (Int,Int) = (rows,cols)
  
  def value() : Array[Svector] = this.values
  
  def getRow( row : Int ) : Svector = {
    require( row <= this.rows && row > 0 ,"Rows over!!  == From Smatrix.getRow.")
    this.values(row-1)
  }
  
  def getElem( row : Int, col : Int ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.getElem.")
    this.values(row-1)(col-1)
  }
  
  def updateElem( row : Int, col : Int, value : Double ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.updateElem.")
    this.values(row-1)(col-1) = value
    this.values(row-1)(col-1)
  }
  
  
}

class SAutoEncoder(
  private var Ws : Array[Smatrix], 
  private var Bs : Svector, 
  private var config : NNConfig
  
  ) extends Serializable {

  def getConf() : NNConfig = this.config
  
  def getW() : Array[Smatrix] = this.Ws
  
  def this(config : NNConfig){
    this(null,null,config)
    InitWeight(config)
  }/// def this(config : NNConfig) 
  
  def InitWeight(config : NNConfig){
    val arr = config.size
    this.Ws = new Array[Smatrix](arr.size -1)
    this.Bs = new Svector (arr.size)
    var t = (0 until arr.size-1 ).map{
      i =>
        this.Ws(i) = new Smatrix( arr(i+1).toInt, (arr(i)+1).toInt, System.currentTimeMillis() )
        this.Bs(i) = 1.0 
    }
  }/// InitWeight !

  def Init(){
    if(this.config != null) InitWeight(this.config)
  }
  
  def getEncode( inData : Sdocs, resPath : String ) = {
    val res = inData.map{
      inDoc =>
        (inDoc.head._2,encode(inDoc))
    }
    import scala.io.Source
    res    
  }
  
  def SGDtrain(inData : Sdocs){
    val start = System.currentTimeMillis
        (0 until this.config.iteration).map{
          j => 
    val res = inData.map { 
      inDoc => 
            print(s"Iteration : $j ,\t")
            NNBP(inDoc)
        } 
    }
    val end = System.currentTimeMillis
    println(s"Training Time is ${end - start} ms ")
  }/// train
  
  def encode( row : Svector ) : Svector = Sigmoid(this.Ws.head.netMulRow(row)) /// encode!
  
  def encode( doc : Sdoc ) : Svector = Sigmoid(this.Ws.head.netMulDoc(doc)) /// encode!
  
  def decode( row : Svector ) : Svector = Sigmoid(this.Ws.last.netMulRow(row) ) /// decode!
  
  def decode( doc : Sdoc, row : Svector ) : Sdoc = {
    val docCols = doc.map( _._1 )
    val numCols = docCols.size
    val docID = doc.head._2
    require( docID == doc.last._2 , "doc Data Error !! == From SAutoEncoder.decode(doc,row)")
    val w1 = this.Ws.last.value()
    val sigIn = (0 until numCols).map{
      i =>
        var sum = 0.0 
        (0 until row.size).map{
          j =>
            sum += w1(docCols(i)-1)(j) * row(j)
        }
        sum += w1(docCols(i)-1).last
        sum
    }.toArray
    val sigOut = Sigmoid(sigIn)
    val res = (0 until numCols).map( i => (docCols(i),docID,sigOut(i)) ).toArray
    res
  } ///decode doc!
  
  def NNFF( doc : Sdoc ) : ( Sdoc, Svector ) ={
    val hiddenOut = encode(doc)
    val endOut = decode(doc,hiddenOut)
    (endOut, hiddenOut)
  } /// NNFF
  
  def NNBP( inDoc : Sdoc ){
    val (endOut,hiddenOut) = NNFF( inDoc )
    val errOutDoc = CalcErr( inDoc, endOut )
    val errHid = CalcErrHidden(errOutDoc, endOut, hiddenOut)
    UpdateW( inDoc, hiddenOut, errHid, errOutDoc)
    
  }/// NNBP
  
  def CalcErr( target : Sdoc, out : Sdoc) : Sdoc = {
    var sum = 0.0 
    /// (t-o)*(1-o)*o
    require( target.size == out.size, "Bad Dimision!!  == From SAutoEncoder.CalcErr")
    target.sortBy { x => x._1 }
    out.sortBy { x => x._1 }
    val minus = (0 until out.size).map{ 
      i =>
        require( target(i)._1 == out(i)._1, "Bad column!! == From SAutoEncoder.CalcErr")
        (target(i)._3 - out(i)._3)
    }
    val loss = minus.map{ x => x * x }.sum / minus.size 
    println(s"row : ${out(1)._2} ,\tloss : $loss , \thalfLoss : ${loss/2}")
    val sigdiver = out.map{
      x =>
        /// outputValue => oVal
        val oVal = x._3
        val res = oVal * (1 - oVal)
//        (x._1, res)
        res
    }
    val errOutDoc = (0 until out.size).map{
      i =>
        (out(i)._1, out(i)._2, minus(i) * sigdiver(i))
    }.toArray
    errOutDoc
  }/// CalcErr
  
  def CalcErrHidden( errOutDoc : Sdoc, out : Sdoc, hiddenOut : Svector ) : Svector ={
    var sum = 0.0
    /// Error out => eo
    /// sum(k) -> eh(k) += eo(h) * w(h,k)
    /// Error Hidden => eh; z = hiddenOut(i)
    /// eh(i) = eh(i) * z * (1-z)
    val w = this.Ws.last
    val err = errOutDoc.sortBy { x => x._1 }.map{ i => i._3 }.toArray
    val errLen = err.size
    val hideLen = hiddenOut.size
    val errHid = new Svector(hideLen)
//    for( i <- 0 until errLen){
//      for( j <- 0 until hideLen){
//        errHid(j) += err(i) * w.getElem(out(i)._1, j.toInt)
//      }
//    }
    (0 until errLen).map{
      i =>
        (0 until hideLen).map{
          j =>
            errHid(j) += err(i) * w.getElem(out(i)._1, (j+1).toInt)
        }
    }
    
//    for( i <- 0 until hideLen){
//      val z = hiddenOut(i) ///out(i)._3  
//      errHid(i) *= z * ( 1 - z )
//    }
    (0 until hideLen).map{
      i =>
        val z = hiddenOut(i)
        errHid(i) *= z * ( 1 - z )
    }
    errHid
  }/// CalcErrHiden
  
  def UpdateW( inDoc : Sdoc, hideOut : Svector, errHid : Svector, errOutDoc : Sdoc){
    val lr = this.config.learningRate
    val hiddenSize = hideOut.size
    val outSize = errOutDoc.size
    /// output layer : deltaW(j,i) = lr * eo(j) * ho(i)
//    for (j <- 0 to config.K - 1) {
//      Bin(j) -= config.stepLength * (errorHidden(j) * alpha + Bin(j) * beta)
//    }
//    for (i <- 0 to Xin.length - 1) {
//      Bout(Xin(i)._1) -= config.stepLength * (errorOut(i) * alpha + Bout(Xin(i)._1) * beta)
//      for (j <- 0 to config.K - 1) {
//        Win(Xin(i)._1)(j) -= 
//          config.stepLength * (errorHidden(j) * Xin(i)._2 * alpha + Win(Xin(i)._1)(j) * beta)
//        Wout(Xin(i)._1)(j) -= 
//          config.stepLength * (errorOut(i) * Xhidden(j) * alpha + Win(Xin(i)._1)(j) * beta)
//      }
//    }
    
    /**
    for( j <- 0 until outSize){
      for( i <- 0 until hiddenSize){
        val k = errOutDoc(j)._1
        this.Ws.last.updateElem(k, i.toInt, 
            ( this.Ws.last.getElem(k, i.toInt) - (lr * errOutDoc(j)._3 * hideOut(i)).toDouble ))
        this.Ws.head.updateElem(i.toInt, k, 
            ( this.Ws.head.getElem(i.toInt, k) - (lr * errHid(i) * inDoc(j)._3 ).toDouble ))
            
      }
    }/// Wout update --> W both update
    **/
    
    ( 0 until outSize ).map{
      j =>
        (0 until hiddenSize).map{
          i =>
            val k = errOutDoc(j)._1
            this.Ws.last.updateElem(k, (i+1).toInt, 
                ( this.Ws.last.getElem(k, (i+1).toInt) + (lr * errOutDoc(j)._3 * hideOut(i)).toDouble ))
            this.Ws.head.updateElem((i+1).toInt, k, 
                ( this.Ws.head.getElem((i+1).toInt, k) + (lr * errHid(i) * inDoc(j)._3 ).toDouble ))
        }
    }
    
    //    for( j <- 0 until hiddenSize){
    //      for( i <- 0 until outSize){
    //        val k = inDoc(i)._1
    //        this.Ws.head.updateElem(j.toInt, k,
    //            (this.Ws.head.getElem(j.toInt, k) + ( lr * errHid(j) * inDoc(i)._3 ).toDouble ))
    //      }
    //    }/// Whide update
    
  }/// UpdateW
  
}/// SAutoEncoder


var count = 0L
val data = sc.textFile("hdfs:///user/hadoop/suanec/Train.data")
// val data = sc.textFile("file:///home/hadoop/suanec/data/MTrick/Train.data")
// count = data.count
val dataSplits = data.map{
  line =>
    val arrt = line.split(',').map( _.toDouble )
    (arrt(1),(arrt(0),arrt(2)))
}
val docs = dataSplits.groupByKey//.sortBy( x => x )
count = docs.count

val ds = readMTrick("hdfs:///user/hadoop/suanec/Train.data")
val dsv = ds.values

val conf = new NNConfig(Array(ds.cols,80,ds.cols),3)
val ae = new SAutoEncoder(conf)
ae.SGDtrain(dsv)










import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

val labels = sc.textFile("hdfs:///user/hadoop/suanec/Train.label").map(x => x.toInt).collect

val enc = ae.getEncode(dsv,"")

val labeledPoint = enc.map( x => LabeledPoint( labels(x._1-1)-1, Vectors.dense(x._2)))
val rdd = sc.parallelize(labeledPoint) 
// val svmData = sc.parallelize(points)
val svmSplits = rdd.randomSplit(Array(0.9,0.1),System.currentTimeMillis)
val svmTrain = svmSplits.head
val svmTest = svmSplits.last
val numIter = 20
val model = SVMWithSGD.train(svmTrain,numIter)

val labelAndPreds = svmTest.collect.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).size.toDouble / svmTest.count
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("Precision = " + trainPrecision )



val dst = readMTrick("hdfs:///user/hadoop/suanec/Test.data")
val dstv = dst.values

val labelst = sc.textFile("hdfs:///user/hadoop/suanec/Test.label").map(x => x.toInt).collect

val enct = ae.getEncode(dstv,"")

val labeledPointt = enc.map( x => LabeledPoint( labels(x._1-1)-1, Vectors.dense(x._2)))
val rdd = sc.parallelize(labeledPointt) 
val labelAndPreds = labeledPointt.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction) } 
val trainErr = labelAndPreds.filter( r => r._1 != r._2).size.toDouble / labeledPointt.size
    println("Training Error = " + trainErr)
    val trainPrecision = 1- trainErr
    println("Precision = " + trainPrecision )

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import org.apache.spark.rdd.RDD
type Svector = Array[Double]
type Sword = (Int,Int,Double)
type SwordBuf = ArrayBuffer[Sword]
type Sdoc = Array[Sword]
type SdocBuf = ArrayBuffer[Sdoc]
type Sdocs = Array[Sdoc]
type Sdata = (Int, Sdoc)
type SDataSet = (Int, Int, RDD[Sdata])
case class DataStats(var rows : Int, var cols : Int, var values : RDD[Sdata]) extends Serializable {
  def setRows( row : Int ) = {
    require(row >= 0, "error row ! == From dataStats.setRows")
    this.rows = row
  }
  def setCols( col : Int ) = {
    require( col >= 0, "error col ! == From dataStats.setCols")
    this.cols = col
  }
  def setValues( value : RDD[Sdata] ) = {
    this.values = value
    setRows(this.values.count.toInt)
    setCols(this.values.map{
      doc => 
        doc._2.last._1
      }.collect.max.toInt
    )
  }
  def this() = this(0,0,null)
  def this(value : RDD[Sdata]) = {
    this(0,0,null)
    setValues(value)

  }
}/// DataStats case class

case class NNConfig(
  var size: Array[Int] = Array(10,2,10),
  var layer: Int = 3,
  var learningRate: Double = 0.1f,
  var iteration : Int = 10,
  activation_function: String = "sigmoid",
  momentum: Double = 0.5f,
  scaling_learningRate: Double = 0.5f,
  weightPenaltyL2: Double = 0.5f,
  nonSparsityPenalty: Double = 0.5f,
  sparsityTarget: Double = 0.2f,
  inputZeroMaskedFraction: Double = 0.8f,
  dropoutFraction: Double = 0.3f,
  testing: Double = 0.1f,
  output_function: String = "sigmoid") extends Serializable {
  def setLearningRate( lr : Double ) = {
    require(lr >= 0, "error lr ! == From NNConfig.setLearningRate")
    this.learningRate = lr
  }

  def setLayer( layer : Int ) = {
    require(layer >= 0, "error layer ! == From NNConfig.setLayer")
    this.layer = layer
  }

  def setSize( size : Array[Int] ) = {
    this.size = size
    setLayer(size.size)
  }

  def setIteration( numIter : Int ) = {
    require( numIter >= 1 , "error numIter ! == From NNConfig.setIteration")
    this.iteration = numIter
  }

}/// NNConfig

def Sigmoid( row : Svector ) : Svector = {
  row.map{
    x => (1 / (1 + Math.exp(-1 * x))).toDouble
  }
}

def softmax(arr : Sdoc) = {
  var sum = 0.0
  var empty = true
  val res = new Sdoc(0)
  arr.foreach{
     i =>
     if(i._3 <= 709) sum += Math.exp(i._3).toDouble
     else empty = false
  }
  if(empty)  (0 until arr.size).map( i => (arr(i)._1, arr(i)._2, (Math.exp(arr(i)._3) /sum).toDouble) ).toArray
  else res.toArray
}

def readMTrick( file : String = "hdfs:///user/hadoop/suanec/Train.data",
                isNorm : Boolean = true, 
                minPartitions : Int = 64 ) : DataStats = {
  val data = sc.textFile( file, minPartitions ).map{
    line =>
      val splits = line.split(',')
      (splits(0).toInt, splits(1).toInt, splits(2).toDouble)
  }  
  val cols = data.max._1.toInt
  val rows = data.max()(Ordering.by[(Int,Int,Double),Int](_._2))._2.toInt

  val docs = data.map( x => (x._2,(x._1, x._2, x._3)) ).groupByKey 
  val normedDocs = docs.map{
    doc => 
      val words = if(isNorm){ softmax(doc._2.toArray) } else doc._2.toArray
      (doc._1, words)
  }.filter( doc => doc._2.size != 0 )
  new DataStats(rows, cols, normedDocs)
}/// readMTricks return dataStats



class Smatrix(
    private var rows : Int, 
    private var cols : Int, 
    private var values : Array[Svector], 
    private var rand : Random
    ) extends Serializable {
  def this() = this(0,0,null,new Random(134L))
  
  def this( values : Array[Svector] ){
    this( values.size.toInt, 0, values, null)
    this.rows = values.size.toInt
    values.map{
      i =>
        if( cols < i.size.toInt ) cols = i.size.toInt
    }
    this.values = new Array[Svector](rows)      
    (0 until values.size).map{
      i => 
        this.values(i) = new Svector(cols)
        values(i).copyToArray(this.values(i),0)
    }
  }///  def Smatrix( values : Array[Svector] )

  def this( rows : Int, cols : Int ){
    this( rows, cols, null, new Random(System.currentTimeMillis()) )
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
    } 
  }/// Smatrix zero
  
  def this( rows : Int, cols : Int, seed : Long ) = {
    this( rows, cols, null, new Random(seed))
    this.values = new Array[Svector](rows)
    (0 until this.values.size).map{
      i => 
        this.values(i) = new Array[Double](cols)
        (0 until this.values(i).size).map{
          j => this.values(i)(j) = (rand.nextDouble()%0.8f) + 0.1f - 0.3f /// -0.2 ~ 0.6
        }
    }
  }/// Smatrix random

  def mulRow( row : Svector ) : Svector = {
    require( row.size == this.cols , "Dimision fault! --From Smatrix.mulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij
    }
    res
  }/// mulRow
  
  def netMulRow( row : Svector ) : Svector = {
    require( (row.size + 1) == this.cols , "Dimision fault! --From Smatrix.netMulRow")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until this.cols -1 ).map{
          j =>
            aij += this.values(i)(j) * row(j) 
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// netMulRow
  
  def mulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols >= doc.last._1, "Dimision fault! --From Smatrix.mulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij
    }
    res
  }/// mulDoc
  
  def netMulDoc( doc : Sdoc ) : Svector = {
    doc.sortWith(_._1 < _._1)
    require( this.cols+1 >= doc.last._1 + 1, "Dimision fault! --From Smatrix.netMulDoc")
    val res = new Svector(this.rows)
    val t = (0 until this.rows).map{
      i =>
        var aij = 0.0 
        (0 until doc.size).map{
          j => 
            val tCol = doc(j)._1 - 1
            aij += this.values(i)(tCol) * doc(j)._3
        }
        res(i) = aij + this.values(i).last
    }
    res
  }/// mulDoc
  
  def size() : (Int,Int) = (rows,cols)
  
  def value() : Array[Svector] = this.values
  
  def getRow( row : Int ) : Svector = {
    require( row <= this.rows && row > 0 ,"Rows over!!  == From Smatrix.getRow.")
    this.values(row-1)
  }
  
  def getElem( row : Int, col : Int ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.getElem.")
    this.values(row-1)(col-1)
  }
  
  def updateElem( row : Int, col : Int, value : Double ) : Double = {
    require( row <= this.rows && col <= this.cols && row > 0 && col > 0, "No Index in Smatrix!! == From Smatrix.updateElem.")
    this.values(row-1)(col-1) = value
    this.values(row-1)(col-1)
  }/// updateElem
  
  
}/// Smatrix


class SAutoEncoder(
  private var Ws : Array[Smatrix], 
  private var Bs : Svector, 
  private var config : NNConfig
  
  ) extends Serializable {

  def getConf() : NNConfig = this.config

  def setLearningRate( lr : Double ) = this.config.setLearningRate(lr)
  
  def getW() : Array[Smatrix] = this.Ws
  
  def this(config : NNConfig){
    this(null,null,config)
    InitWeight(config)
  }/// def this(config : NNConfig) 
  
  def InitWeight(config : NNConfig){
    val arr = config.size
    this.Ws = new Array[Smatrix](arr.size -1)
    this.Bs = new Svector (arr.size)
    var t = (0 until arr.size-1 ).map{
      i =>
        this.Ws(i) = new Smatrix( arr(i+1).toInt, (arr(i)+1).toInt, System.currentTimeMillis() )
        this.Bs(i) = 1.0 
    }
  }/// InitWeight !

  def Init() = if(this.config != null) InitWeight(this.config)
  
  def getEncode( inData : Sdocs, resPath : String ) = {
    val res = inData.map{
      inDoc =>
        (inDoc.head._2,encode(inDoc))
    }
    import scala.io.Source
    res    
  }
  
  def SGDtrain(inData : Sdocs){
    val start = System.currentTimeMillis
      (0 until this.config.iteration).map{
        j => 
          val res = inData.map { 
            inDoc => 
              print(s"Iteration : $j ,\t")
              NNBP(inDoc)
          } 
      } 
    val end = System.currentTimeMillis
    println(s"Training Time is ${end - start} ms ")
  }/// train

  // def SGDtrain(inData : Sdata){
  //   val start = System.currentTimeMillis
  //       (0 until this.config.iteration).map{
  //         j => 
  //           print(s"Iteration : $j ,\t")
  //           NNBP(inData._2)
  //       } 
  //   }
  //   val end = System.currentTimeMillis
  //   println(s"Training Time is ${end - start} ms ")
  // }/// train
  
  def encode( row : Svector ) : Svector = Sigmoid(this.Ws.head.netMulRow(row)) /// encode!
  
  def encode( doc : Sdoc ) : Svector = Sigmoid(this.Ws.head.netMulDoc(doc)) /// encode!
  
  def decode( row : Svector ) : Svector = Sigmoid(this.Ws.last.netMulRow(row) ) /// decode!
  
  def decode( doc : Sdoc, row : Svector ) : Sdoc = {
    val docCols = doc.map( _._1 )
    val numCols = docCols.size
    val docID = doc.head._2
    require( docID == doc.last._2 , "doc Data Error !! == From SAutoEncoder.decode(doc,row)")
    val w1 = this.Ws.last.value()
    val sigIn = (0 until numCols).map{
      i =>
        var sum = 0.0 
        (0 until row.size).map{
          j =>
            sum += w1(docCols(i)-1)(j) * row(j)
        }
        sum += w1(docCols(i)-1).last
        sum
    }.toArray
    val sigOut = Sigmoid(sigIn)
    val res = (0 until numCols).map( i => (docCols(i),docID,sigOut(i)) ).toArray
    res
  } ///decode doc!
  
  def NNFF( doc : Sdoc ) : ( Sdoc, Svector ) ={
    val hiddenOut = encode(doc)
    val endOut = decode(doc,hiddenOut)
    (endOut, hiddenOut)
  } /// NNFF
  
  def NNBP( inDoc : Sdoc ){
    val (endOut,hiddenOut) = NNFF( inDoc )
    val errOutDoc = CalcErr( inDoc, endOut )
    val errHid = CalcErrHidden(errOutDoc, endOut, hiddenOut)
    UpdateW( inDoc, hiddenOut, errHid, errOutDoc)
    
  }/// NNBP
  
  def CalcErr( target : Sdoc, out : Sdoc) : Sdoc = {
    var sum = 0.0 
    /// (t-o)*(1-o)*o
    require( target.size == out.size, "Bad Dimision!!  == From SAutoEncoder.CalcErr")
    target.sortBy { x => x._1 }
    out.sortBy { x => x._1 }
    val minus = (0 until out.size).map{ 
      i =>
        require( target(i)._1 == out(i)._1, "Bad column!! == From SAutoEncoder.CalcErr")
        (target(i)._3 - out(i)._3)
    }
    val loss = minus.map{ x => x * x }.sum / minus.size 
    println(s"row : ${out(1)._2} ,\tloss : $loss , \thalfLoss : ${loss/2}")
    val sigdiver = out.map{
      x =>
        /// outputValue => oVal
        val oVal = x._3
        val res = oVal * (1 - oVal)
        res
    }
    val errOutDoc = (0 until out.size).map{
      i =>
        (out(i)._1, out(i)._2, minus(i) * sigdiver(i))
    }.toArray
    errOutDoc
  }/// CalcErr
  
  def CalcErrHidden( errOutDoc : Sdoc, out : Sdoc, hiddenOut : Svector ) : Svector ={
    var sum = 0.0
    /// Error out => eo
    /// sum(k) -> eh(k) += eo(h) * w(h,k)
    /// Error Hidden => eh; z = hiddenOut(i)
    /// eh(i) = eh(i) * z * (1-z)
    val w = this.Ws.last
    val err = errOutDoc.sortBy { x => x._1 }.map{ i => i._3 }.toArray
    val errLen = err.size
    val hideLen = hiddenOut.size
    val errHid = new Svector(hideLen)
    (0 until errLen).map{
      i =>
        (0 until hideLen).map{
          j =>
            errHid(j) += err(i) * w.getElem(out(i)._1, (j+1).toInt)
        }
    }
    
    (0 until hideLen).map{
      i =>
        val z = hiddenOut(i)
        errHid(i) *= z * ( 1 - z )
    }
    errHid
  }/// CalcErrHiden
  
  def UpdateW( inDoc : Sdoc, hideOut : Svector, errHid : Svector, errOutDoc : Sdoc){
    val lr = this.config.learningRate
    val hiddenSize = hideOut.size
    val outSize = errOutDoc.size
    /// output layer : deltaW(j,i) = lr * eo(j) * ho(i)
    ( 0 until outSize ).map{
      j =>
        (0 until hiddenSize).map{
          i =>
            val k = errOutDoc(j)._1
            this.Ws.last.updateElem(k, (i+1).toInt, 
                ( this.Ws.last.getElem(k, (i+1).toInt) + (lr * errOutDoc(j)._3 * hideOut(i)).toDouble ))
            this.Ws.head.updateElem((i+1).toInt, k, 
                ( this.Ws.head.getElem((i+1).toInt, k) + (lr * errHid(i) * inDoc(j)._3 ).toDouble ))
        }
    }
  }/// UpdateW
  
}/// SAutoEncoder



-----------------------------------------------------------------------------
val start = System.currentTimeMillis
val total = 10000    * 10000
(0 until total).map( x => x + 1 )
val end = System.currentTimeMillis
val diff = end - start
val per = (diff/total.toDouble*1000).toString + " millis"

val ds = readMTrick()
val docs = ds.values
val doc = docs.first
val conf = new NNConfig
conf.setSize(Array(ds.cols,80,ds.cols))
conf.setLearningRate(0.4)
val ae = new SAutoEncoder (conf)
(0 until 5).map( j => (0 until 10001).map( i => ae.NNBP(doc) ) )
(0 until 10001).map( i => ae.NNBP(doc) )
val docr = ae.decode(doc, ae.encode(doc))
(0 until doc.size).map( i => println(s"doc : ${doc(i)._3}, \tdocr : ${docr(i)._3}, \t diff : ${(doc(i)._3 - docr(i)._3)/doc(i)._3};"))
-----------------------------------------------------------------------------
经历了那么多坎坷磨难煎熬之后 
我最怕的是 
不是你
天下事有难易乎？为之，则难者亦易矣；不为，则易者亦难矣。 --《为学》

-----------------------------------------------------------------------------
