spark-submit --class test.test --executor-memory 20G --driver-memory 20G --master local[4] --conf spark.kryoserializer.buffer.max.mb=800 --num-executors 6 /home/hadoop/fhtn/MAEyarn.jar 4 20000 61188 18800 800 /home/hadoop/fhtn/ /home/hadoop/fhtn/MAEres/ datafile.data

--master: yarn local local[]
--conf spark.kryoserializer.buffer.max.mb=800 视数据大小而定

4:nWorker
2000:nDocset(每个worker上的样本数量)
61188:M
18800:N
800:K
输入文件路径
输出文件路径
输入文件名
