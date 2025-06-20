from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib
import os

class SimpleMonitor13(switch.SimpleSwitch13):

    # def __init__(self, *args, **kwargs):

    #     super(SimpleMonitor13, self).__init__(*args, **kwargs)
    #     self.datapaths = {}
    #     self.monitor_thread = hub.spawn(self._monitor)

    #     start = datetime.now()

    #     self.flow_training()

    #     end = datetime.now()
    #     print("Training time: ", (end-start))
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        # self.last_predict_timestamp = 0
        self.blocked_ips = set()


        start = datetime.now()

        model_path = 'best_model.pkl'

        if os.path.exists(model_path):
            self.flow_model = joblib.load(model_path)
            self.logger.info("Loaded model from {}".format(model_path))
        else:
            self.flow_training()

        end = datetime.now()
        print("Model loading/training time: ", (end - start))


    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(2)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()


        file0 = open("PredictFlowStatsfile.csv","a")
        # file0.write('timestamp,tp_dst,ip_proto,total_duration_msec,packet_count,byte_count,packet_count_per_second\n')
        body = ev.msg.body
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
        # for stat in body:
            # print(stat)
            ip_proto = 0
            ip_src = stat.match['ipv4_src']
            # print(ip_src)
            ip_dst = stat.match['ipv4_dst']
            try:
                if stat.match['ip_proto'] == 6:
                    ip_proto = 6
                    tp_dst = stat.match['tcp_dst']

                elif stat.match['ip_proto'] == 17:
                    ip_proto = 17
                    tp_dst = stat.match['udp_dst']
            except:
                pass

            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
            except:
                packet_count_per_second = 0
            
            total_duration_msec = stat.duration_sec * 1000 + (stat.duration_nsec / 1e6)
                
            file0.write("{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, str(ip_src), str(ip_dst),
                        tp_dst,
                        ip_proto,
                        total_duration_msec,
                        # stat.idle_timeout, stat.hard_timeout,
                        stat.packet_count,stat.byte_count, # sum from fwd and bwd
                        packet_count_per_second
                        ))
            # test_data = [int(tp_dst), ip_src, ip_dst, int(ip_proto), total_duration_msec, int(stat.packet_count),int(stat.byte_count), packet_count_per_second]
            # test_data_reshaped = np.array(test_data).reshape(1, -1)
            # print(test_data)
            # print(self.flow_model.predict(test_data_reshaped))
            
        file0.close()

    def flow_training(self):

        self.logger.info("Flow Training ...")

        flow_dataset = pd.read_csv('dataset.csv')

        X_flow = flow_dataset.iloc[:, 1:-1].values
        X_flow = X_flow.astype('float64')

        y_flow = flow_dataset.iloc[:, -1].values

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)

        classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
        self.flow_model = classifier.fit(X_flow_train, y_flow_train)

        y_flow_pred = self.flow_model.predict(X_flow_test)

        self.logger.info("------------------------------------------------------------------------------")

        self.logger.info("confusion matrix")
        cm = confusion_matrix(y_flow_test, y_flow_pred)
        self.logger.info(cm)

        acc = accuracy_score(y_flow_test, y_flow_pred)

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

        model_path = 'best_model.pkl'
        joblib.dump(self.flow_model, model_path)
        self.logger.info("Model saved to {}".format(model_path))

    def limit_ip_rate(self, ip, datapath, rate=1000):
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # Meter ID để sử dụng, ví dụ 1 hoặc sinh động hơn
        meter_id = 1

        # Gửi meter mod để giới hạn băng thông (pktps = packets per second)
        bands = [parser.OFPMeterBandDrop(rate=rate, burst_size=rate)]
        meter_mod = parser.OFPMeterMod(
            datapath=datapath,
            command=ofproto.OFPMC_ADD,
            flags=ofproto.OFPMF_KBPS,  # hoặc OFPMF_PKTPS nếu muốn giới hạn theo packets
            meter_id=meter_id,
            bands=bands
        )
        datapath.send_msg(meter_mod)

        # Tạo match rule cho IP attacker
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip)

        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst = [
            parser.OFPInstructionMeter(meter_id, ofproto.OFPIT_METER),
            parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
        ]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=90,  # thấp hơn block nhưng cao hơn default
            match=match,
            instructions=inst,
            idle_timeout=20,  # tuỳ chỉnh
            hard_timeout=0
        )
        datapath.send_msg(mod)

        self.logger.info(f"[ RATE LIMIT] IP {ip} is rate limited to {rate} KBps on switch {datapath.id}")


    # def block_ip(self, ip, datapath):
    #     parser = datapath.ofproto_parser
    #     ofproto = datapath.ofproto

    #     match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip)
    #     actions = []  # không có hành động → drop

    #     inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

    #     mod = parser.OFPFlowMod(
    #         datapath=datapath,
    #         priority=100,  # Ưu tiên cao hơn flow mặc định
    #         match=match,
    #         instructions=inst,
    #         idle_timeout=60,  # sau 60s sẽ tự xóa (tuỳ chỉnh)
    #         hard_timeout=0
    #     )
    #     datapath.send_msg(mod)

    #     self.logger.info(f"[ BLOCK] IP {ip} has been blocked on switch {datapath.id}")


    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')

            if predict_flow_dataset.shape[0] < 5:
                return  # Dữ liệu chưa đủ


            predict_flow_dataset = predict_flow_dataset.tail(1000)

            X_predict_flow = predict_flow_dataset.iloc[:, 3:].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            y_flow_pred = self.flow_model.predict(X_predict_flow)

            legitimate_trafic = 0
            ddos_trafic = 0

            victim_count = {}

            # for i in y_flow_pred:
            #     if i == 0:
            #         legitimate_trafic = legitimate_trafic + 1
            #     else:
            #         ddos_trafic = ddos_trafic + 1
            #         victim = predict_flow_dataset.iloc[i, 1]
            #         if victim not in victim_count:
            #             victim_count[victim] = 1 
            #         else:
            #             victim_count[victim] += 1
            # victim = max(victim_count, key=victim_count.get)

            label_map = {
                0: "Benign",
                1: "DDoS"
            }

            found_labels = set()

            attacker_count = {}

            for idx, pred_label in enumerate(y_flow_pred):
                label_name = label_map.get(pred_label, "Unknown")
                found_labels.add(label_name)

                if pred_label == 0:
                    legitimate_trafic += 1
                else:
                    ddos_trafic += 1

                    victim = predict_flow_dataset.iloc[idx, 2]  # ip_dst
                    if victim not in victim_count:
                        victim_count[victim] = 1 
                    else:
                        victim_count[victim] += 1

                    attacker = predict_flow_dataset.iloc[idx, 1]  # ip_src
                    if attacker not in attacker_count:
                        attacker_count[attacker] = 1
                    else:
                        attacker_count[attacker] += 1

            victim = max(victim_count, key=victim_count.get)
            attacker = max(attacker_count, key=attacker_count.get)


            # print(legitimate_trafic, ddos_trafic)
                    

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic/len(y_flow_pred)*100) > 80 or ddos_trafic < 40:
                self.logger.info("legitimate trafic ...")
            else:
                self.logger.info("ddos trafic ...")
                self.logger.info("Labels found: {}".format(list(found_labels - {"Benign"})))
                self.logger.info("victim ip {}".format(victim))
                self.logger.info("attacker ip: {}".format(attacker))

                for dp in self.datapaths.values():
                    self.limit_ip_rate(attacker, dp, rate=100)


            self.logger.info("------------------------------------------------------------------------------")
            
            file0 = open("PredictFlowStatsfile.csv","w")
            
            file0.write('timestamp,ip_src,ip_dst,tp_dst,ip_proto,total_duration_msec,packet_count,byte_count,packet_count_per_second\n')
            file0.close()
        except:
            file0 = open("PredictFlowStatsfile.csv","w")
            
            file0.write('timestamp,ip_src,ip_dst,tp_dst,ip_proto,total_duration_msec,packet_count,byte_count,packet_count_per_second\n')
            file0.close()
