import pandas as pd

test_data = pd.read_csv("./Test.csv")

del test_data["creationDate"]
del test_data["symptom"]
del test_data["hostName"]
del test_data["ticketType"]
del test_data["summary"]
#del test_data["mgmtIpAddress"]

test_data.loc[test_data["serviceType"] == "Network Carrier Alert Resolution", "serviceType"] = 1
test_data.loc[test_data["serviceType"] == "Network Hardware Alert Resolution", "serviceType"] = 2
test_data.loc[test_data["serviceType"] == "Network Security Alert Resolution", "serviceType"] = 3
test_data.loc[test_data["serviceType"] == "Network Configuration Alert Resolution", "serviceType"] = 4
test_data.loc[test_data["serviceType"] == "Network Capacity/Performance Alert Resolution", "serviceType"] = 5
test_data.loc[test_data["serviceType"] == "Network General Alert Resolution", "serviceType"] = 6

test_data["site"] = test_data["site"].fillna(0)

test_data.loc[test_data["site"] == " ", "site"] = 0
test_data.loc[test_data["site"] == "NEWYORK", "site"] = 1
test_data.loc[test_data["site"] == "LONDON", "site"] = 2
test_data.loc[test_data["site"] == "TOKYO", "site"] = 3

#test_data["ticketType"] = test_data["ticketType"].fillna(0)
#test_data.loc[test_data["ticketType"] == " ", "ticketType"] = 0
#test_data.loc[test_data["ticketType"] == 20001.0, "ticketType"] = 20001

test_data.loc[test_data["DayofWeek "] == "Mon", "DayofWeek "] = 1
test_data.loc[test_data["DayofWeek "] == "Tue", "DayofWeek "] = 2
test_data.loc[test_data["DayofWeek "] == "Wed", "DayofWeek "] = 3
test_data.loc[test_data["DayofWeek "] == "Thu", "DayofWeek "] = 4
test_data.loc[test_data["DayofWeek "] == "Fri", "DayofWeek "] = 5

print(test_data.head(5))

test_data.to_csv('./test_simple.csv', index = False, header=True)