import pandas as pd
import sys

data = pd.read_csv('/home/manoj/Downloads/skype/test1.csv')
matched_data = pd.read_csv('/home/manoj/Downloads/skype/RBCMRBCM20200527-blocks.csv')

columns = data.columns.tolist()
number_of_rows = len(data.index)
new_values = []
print('Processing')

for index, row in data.iterrows():
    matched = False
    for indx2, value in matched_data.iterrows():
        if row['Parent_Order_ID2'].split('-')[0] == value['PARENT ORDER ID']:
            data.loc[index, ['ACCOUNT2']] = value['ACCOUNT']
            if int(row['Order_Quantity2']) != 0:
                Order_Quantity2 = (value['ALLOCATED QUANTITY'] / row['Order_Quantity2']) * row['Order_Quantity2']
                ACTION_QUANTITY2 = (value['ALLOCATED QUANTITY'] / row['Order_Quantity2']) * row['ACTION_QUANTITY2']
                ROUTE_QTY2 = (value['ALLOCATED QUANTITY'] / row['Order_Quantity2']) * row['ROUTE_QTY2']
            print(Order_Quantity2, ACTION_QUANTITY2, ROUTE_QTY2)


            data.loc[index, ['Order_Quantity2']] = Order_Quantity2
            data.loc[index, ['ACTION_QUANTITY2']] = Order_Quantity2
            data.loc[index, ['ROUTE_QTY2']] = Order_Quantity2
            new_values.append(data.values[index])
            matched = True
            print(data['Order_Quantity2'],data['ACTION_QUANTITY2'],data['ROUTE_QTY2'])
    if not matched:
        new_values.append(data.values[index])
    # print
    # 'total Records Processed {}/{}'.format(index + 1, number_of_rows), '\r',

print('\nProcessing Completed')


df = pd.DataFrame(new_values, columns=columns)

# saving the dataframe
df.to_csv('output.csv', index=False)
print('file Sucessfully Saved')




