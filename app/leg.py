main_list = [11, 21, 19, 18, 46]
check_elements = [11, 19, 46]

k = all([item in check_elements  for item in main_list])
print(k)
