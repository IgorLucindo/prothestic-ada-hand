class CurrentObject:
    def __init__(self, dict_objects, classes, focal_length):
        self.dict_objects = dict_objects
        self.classes = classes
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
        boxes = results[0]['boxes']
        labels = results[0]['labels']
        scores = results[0]['scores']

        # choose current object
        self.score = 0
        for i in range(len(labels)):
            if self.classes[labels[i].item()] not in self.dict_objects:
                continue

            if scores[i].item() > max(self.score, 0.25):
                # set current object atributes
                self.detected = True
                self.score = scores[i].item()
                self.name = self.classes[labels[i].item()]
                self.box = boxes[i]

        if self.detected:
            # set current object atributes
            self.time = 0
            dict_obj = self.dict_objects[self.name]
            self.grasp = dict_obj['grasp']
            width = int(self.box[2].item()) - int(self.box[0].item())
            self.dist= (dict_obj['width'] * self.focal_length)/width
            self.vel = -(self.dist - self.prev_dist)/deltaTime
        else:
            # reset current object atributes
            self.resetObject(deltaTime, resetGraspTimer)

    # set current object previous atributes
    def setPrevObject(self):
        self.prev_grasp = self.grasp
        self.prev_dist = self.dist