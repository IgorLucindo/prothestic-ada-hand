class CurrentObject:
    def __init__(self, dict_objects, focal_length):
        self.dict_objects = dict_objects
        self.focal_length = focal_length

        self.name = "nothing"
        self.grasp = "None"
        self.prev_grasp = "None"
        self.score = 0
        self.box = []
        self.dist = 0
        self.prev_dist = 0
        self.vel = 0

        self.detected = False
        self.time = 0

    # reset current object atributes
    def resetObject(self, deltaTime, resetGraspTimer):
        self.name = "nothing"
        if self.time < resetGraspTimer: self.time += deltaTime
        else:
            self.time = 0
            self.grasp = "None"
        self.score = 0
        self.box = []
        self.dist= 0
        self.vel = 0

    # choose the object with highest score
    def setObject(self, results, deltaTime, resetGraspTimer):
        self.detected = False

        # get atributes
        boxes = results.boxes[0]
        labels = results.labels[0]
        scores = results.scores[0]

        # choose current object
        self.score = 0
        for i in range(len(labels)):
            print(labels[i])
            if labels[i] not in self.dict_objects:
                continue

            if scores[i] > max(self.score, 0.25):
                # set current object atributes
                self.detected = True
                self.score = scores[i]
                self.name = labels[i]
                self.box = boxes[i]

        if self.detected:
            # set current object atributes
            self.time = 0
            dict_obj = self.dict_objects[self.name]
            self.grasp = dict_obj['grasp']
            width = int(self.box[2]) - int(self.box[0])
            self.dist= (dict_obj['width'] * self.focal_length)/width
            self.vel = -(self.dist - self.prev_dist)/deltaTime
        else:
            # reset current object atributes
            self.resetObject(deltaTime, resetGraspTimer)

    # set current object previous atributes
    def setPrevObject(self):
        self.prev_grasp = self.grasp
        self.prev_dist = self.dist