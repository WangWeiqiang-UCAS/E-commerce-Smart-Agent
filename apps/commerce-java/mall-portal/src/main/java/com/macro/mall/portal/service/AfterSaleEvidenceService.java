package com.macro.mall.portal.service;

import com.macro.mall.portal.domain.AfterSaleEvidence;
import org.springframework.web.multipart.MultipartFile;

public interface AfterSaleEvidenceService {
    AfterSaleEvidence.UploadResponse upload(MultipartFile file);
    void attach(Long afterSaleId, AfterSaleEvidence.AttachRequest request);
    AfterSaleEvidence.DownloadResponse download(Long afterSaleId, String objectKey);
}
