import pandas as pd

training_data = pd.read_csv("./Train.csv")

del training_data["creationDate"]
del training_data["symptom"]
del training_data["hostName"]
del training_data["summary"]
#del training_data["mgmtIpAddress"]

training_data.loc[training_data["serviceType"] == "Network Carrier Alert Resolution", "serviceType"] = 1
training_data.loc[training_data["serviceType"] == "Network Hardware Alert Resolution", "serviceType"] = 2
training_data.loc[training_data["serviceType"] == "Network Security Alert Resolution", "serviceType"] = 3
training_data.loc[training_data["serviceType"] == "Network Configuration Alert Resolution", "serviceType"] = 4
training_data.loc[training_data["serviceType"] == "Network Capacity/Performance Alert Resolution", "serviceType"] = 5
training_data.loc[training_data["serviceType"] == "Network General Alert Resolution", "serviceType"] = 6

training_data["site"] = training_data["site"].fillna(0)

training_data.loc[training_data["site"] == " ", "site"] = 0
training_data.loc[training_data["site"] == "NEWYORK", "site"] = 1
training_data.loc[training_data["site"] == "LONDON", "site"] = 2
training_data.loc[training_data["site"] == "TOKYO", "site"] = 3

training_data.loc[training_data["ticketType"] == " ", "ticketType"] = 0
training_data.loc[training_data["ticketType"] == "20001", "ticketType"] = 20001

training_data.loc[training_data["DayofWeek "] == "Mon", "DayofWeek "] = 1
training_data.loc[training_data["DayofWeek "] == "Tue", "DayofWeek "] = 2
training_data.loc[training_data["DayofWeek "] == "Wed", "DayofWeek "] = 3
training_data.loc[training_data["DayofWeek "] == "Thu", "DayofWeek "] = 4
training_data.loc[training_data["DayofWeek "] == "Fri", "DayofWeek "] = 5

training_data.loc[training_data["Actionability"] == "Non-actionable", "Actionability"] = 0
training_data.loc[training_data["Actionability"] == "Actionable", "Actionability"] = 1

training_data.to_csv('./temp.csv', index = False, header=True)
