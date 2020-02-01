# ELK
cp logstash-udp.conf /usr/share/logstash/bin/
sudo -u elasticsearch JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/ /usr/share/elasticsearch/bin/elasticsearch -d > /data/logs/elasticsearch.log 2>&1
/usr/share/logstash/bin/logstash -f logstash-udp.conf > /data/logs/logstash.log 2>&1 &
/usr/share/kibana/bin/kibana --server.host=0.0.0.0 > /data/logs/kibana.log 2>&1 &