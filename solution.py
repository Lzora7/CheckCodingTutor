def find_max(arr):
    # Проверяем, что массив не пустой
    if not arr:
        raise ValueError("Массив не может быть пустым")
    
    # Берем первый элемент как начальный максимум
    max_value = arr[0]
    
    # Перебираем все элементы начиная со второго
    for num in arr[1:]:
        if num > max_value:
            max_value = num
    
    return max_value
