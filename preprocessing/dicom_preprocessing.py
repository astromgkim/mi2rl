
def dicom_header_copy(dcm, ds):

  ds.ImageType='DERIVED\PRIMARY\AXIAL'#dcm.ImageType
  ds.SOPClassUID=dcm.SOPClassUID
  ds.SOPInstanceUID=dcm.SOPInstanceUID     
  
  try:
      ds.StudyDate=dcm.StudyDate
  except:
      pass
  try:
      ds.SeriesDate=dcm.SeriesDate
  except:
      pass
  try:
      ds.AcquisitionDate=dcm.AcquisitionDate
  except:
      pass
  try:    
      ds.ContentDate=dcm.ContentDate
  except:
      pass
  try:    
      ds.StudyTime=dcm.StudyTime
  except:
      pass
  try:    
      ds.SeriesTime=dcm.SeriesTime
  except:
      pass
  try:    
      ds.AcquisitionTime=dcm.AcquisitionTime
  except:
      pass
  try:    
      ds.ContentTime=dcm.ContentTime
  except:
      pass
  try:    
      ds.AccessionNumber=dcm.AccessionNumber
  except:
      pass
  try:    
      ds.Modality=dcm.Modality
  except:
      pass
  try:    
      ds.Manufacturer=dcm.Manufacturer
  except:
      pass
  try:    
      ds.InstitutionName=dcm.InstitutionName
  except:
      pass
  try:    
      ds.InstitutionAddress=dcm.InstitutionAddress
  except:
      pass
  try:    
      ds.ReferringPhysicianName=dcm.ReferringPhysicianName
  except:
      pass
  try:    
      ds.StationName=dcm.StationName
  except:
      pass
  try:    
      ds.StudyDescription=dcm.StudyDescription
  except:
      pass
  try:    
      ds.SeriesDescription=dcm.SeriesDescription
  except:
      pass

      #ds.PhysiciansOfRecord=dcm.PhysiciansOfRecord
  try:
      ds.PerformingPhysicianName=dcm.PerformingPhysicianName
  except:
      pass
  try:
      ds.OperatorsName=dcm.OperatorsName
  except:
      pass
  try:    
      ds.ManufacturerModelName=dcm.ManufacturerModelName
  except:
      pass
  try:    
      ds.PatientName=dcm.PatientName
  except:
      pass
  try:    
      ds.PatientID=dcm.PatientID
  except:
      pass
  try:    
      ds.PatientBirthDate=dcm.PatientBirthDate
  except:
      pass
  try:    
      ds.PatientSex=dcm.PatientSex
  except:
      pass
      #ds.OtherPatientNames=dcm.OtherPatientNames
  try:
      ds.PatientAge=dcm.PatientAge
  except:
      pass
  try:    
      ds.BodyPartExamined=dcm.BodyPartExamined
  except:
      pass
  try:    
      ds.SliceThickness=dcm.SliceThickness
  except:
      pass
  try:    
      ds.KVP=dcm.KVP
  except:
      pass
  try:    
      ds.DataCollectionDiameter=dcm.DataCollectionDiameter
  except:
      pass
  try:    
      ds.DeviceSerialNumber=dcm.DeviceSerialNumber
  except:
      pass
  try:    
      ds.SoftwareVersions=dcm.SoftwareVersions
  except:
      pass
  try:    
      ds.ProtocolName=dcm.ProtocolName
  except:
      pass
  try:
      ds.ReconstructionDiameter=dcm.ReconstructionDiameter
  except:
      pass
  try:
      ds.DistanceSourceToDetector=dcm.DistanceSourceToDetector
  except:
      pass
  try:
      ds.DistanceSourceToPatient=dcm.DistanceSourceToPatient
  except:
      pass
  try:
      ds.GantryDetectorTilt=dcm.GantryDetectorTilt
  except:
      pass
  try:
      ds.TableHeight=dcm.TableHeight
  except:
      pass
  try:
      ds.RotationDirection=dcm.RotationDirection
  except:
      pass
  try:
      ds.ExposureTime=dcm.ExposureTime
  except:
      pass
  try:
      ds.XRayTubeCurrent=dcm.XRayTubeCurrent
  except:
      pass
  try:
      ds.Exposure=dcm.Exposure
  except:
      pass
  try:
      ds.FilterType=dcm.FilterType
  except:
      pass
  try:
      ds.GeneratorPower=dcm.GeneratorPower
  except:
      pass
  try:
      ds.FocalSpots=dcm.FocalSpots
  except:
      pass
  try:
      ds.DateOfLastCalibration=dcm.DateOfLastCalibration
  except:
      pass
  try:
      ds.TimeOfLastCalibration=dcm.TimeOfLastCalibration
  except:
      pass
  try:
      ds.ConvolutionKernel=dcm.ConvolutionKernel
  except:
      pass
  try:
      ds.PatientPosition=dcm.PatientPosition
  except:
      pass
  try:
      ds.StudyInstanceUID=dcm.StudyInstanceUID
  except:
      pass
  try:
      ds.SeriesInstanceUID=dcm.SeriesInstanceUID
  except:
      pass
  try:
      ds.StudyID=dcm.StudyID
  except:
      pass
  try:
      ds.SeriesNumber=dcm.SeriesNumber
  except:
      pass
  try:
      ds.AcquisitionNumber=dcm.AcquisitionNumber
  except:
      pass
  try:
      ds.InstanceNumber=dcm.InstanceNumber
  except:
      pass
  try:
      ds.ImagePositionPatient=dcm.ImagePositionPatient
  except:
      pass
  try:
      ds.ImageOrientationPatient=dcm.ImageOrientationPatient
  except:
      pass
  try:
      ds.FrameOfReferenceUID=dcm.FrameOfReferenceUID
  except:
      pass
  try:
      ds.PositionReferenceIndicator=dcm.PositionReferenceIndicator
  except:
      pass
  try:
      ds.SliceLocation=dcm.SliceLocation
  except:
      pass
  try:
      ds.ImageComments=dcm.ImageComments
  except:
      pass
  try:
      ds.SamplesPerPixel=1
  except:
      pass
  try:
      ds.PhotometricInterpretation=dcm.PhotometricInterpretation
  except:
      pass
  try:
      ds.Rows=dcm.Rows
  except:
      pass
  try:
      ds.Columns=dcm.Columns
  except:
      pass
  try:
      ds.PixelSpacing=dcm.PixelSpacing
  except:
      pass
  try:
      ds.BitsAllocated=dcm.BitsAllocated
  except:
      pass
  try:
      ds.BitsStored=dcm.BitsStored
  except:
      pass
  try:
      ds.HighBit=dcm.HighBit
  except:
      pass
  try:
      ds.PixelRepresentation=dcm.PixelRepresentation
  except:
      pass
  try:
      ds.WindowCenter=dcm.WindowCenter
  except:
      pass
  try:
      ds.WindowWidth=dcm.WindowWidth
  except:
      pass
  try:
      ds.RescaleIntercept = '-1024'
  except:
      pass
  try:
      ds.RescaleSlope = '1'
  except:
      pass
  try:
      ds.WindowCenterWidthExplanation=dcm.WindowCenterWidthExplanation
  except:
      pass

  return ds
