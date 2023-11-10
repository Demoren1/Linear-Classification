class config:
  def __init__(self, file_path, features, shift):
    self.features = features
    self.shift = shift
    self.data = self.get_data(file_path)
  
  def get_data(self, file_path):
    tmp_data = []

    with open(file_path) as file:
        for line in file:
            splitted = line.split()

            if (len(splitted) < 1):
                break
            
            formatted_data = [0 for x in range(self.features + 1)]
            if float(splitted[0]) - self.shift >= 0:
                formatted_data[0] = 1
            else:
               formatted_data[0] = -1
               
            counter = 1
            for i in range(1, self.features + 1):
        
                if (counter > len(splitted) - 1):
                    break

                word = splitted[counter].split(':')
                if (float(word[0]) != i):
                    continue

                formatted_data[int(word[0])] = float(word[1])
                counter += 1

            tmp_data.append(formatted_data)
    return tmp_data
  
  def show_data(self):
    for line in self.data:
       print(line)

