from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Cow(Base):
    __tablename__ = 'cows'
    id = Column(Integer, primary_key=True)
    tag = Column(String, unique=True, nullable=True)  # Optional human-readable ID
    created_at = Column(DateTime, default=func.now())
    
    # Relationship to tracklets
    tracklets = relationship("Tracklet", back_populates="cow")

    def __repr__(self):
        return f"<Cow(id={self.id}, tag={self.tag})>"

class Tracklet(Base):
    __tablename__ = 'tracklets'
    id = Column(Integer, primary_key=True)
    cow_id = Column(Integer, ForeignKey('cows.id'), nullable=True)
    camera_id = Column(String, nullable=False)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime, nullable=True)
    
    # Relationships
    cow = relationship("Cow", back_populates="tracklets")
    detections = relationship("Detection", back_populates="tracklet", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tracklet(id={self.id}, cow_id={self.cow_id}, camera_id={self.camera_id})>"

class Detection(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    tracklet_id = Column(Integer, ForeignKey('tracklets.id'), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    posture = Column(String, nullable=False)  # "standing" or "lying"
    
    # Relationship
    tracklet = relationship("Tracklet", back_populates="detections")

    def __repr__(self):
        return (f"<Detection(id={self.id}, tracklet_id={self.tracklet_id}, "
                f"posture={self.posture}, timestamp={self.timestamp})>")
