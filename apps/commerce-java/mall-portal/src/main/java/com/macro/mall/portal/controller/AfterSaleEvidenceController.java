package com.macro.mall.portal.controller;

import com.macro.mall.common.api.CommonResult;
import com.macro.mall.portal.domain.AfterSaleEvidence;
import com.macro.mall.portal.service.AfterSaleEvidenceService;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Pattern;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/** Customer-owned after-sales evidence; never accepts a client-provided URL. */
@RestController
@Validated
@RequestMapping("/returnApply/evidence")
public class AfterSaleEvidenceController {
    private final AfterSaleEvidenceService evidenceService;

    public AfterSaleEvidenceController(AfterSaleEvidenceService evidenceService) {
        this.evidenceService = evidenceService;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Upload one private after-sales evidence file")
    public CommonResult<AfterSaleEvidence.UploadResponse> upload(@RequestParam("file") MultipartFile file) {
        return CommonResult.success(evidenceService.upload(file));
    }

    @PostMapping("/{afterSaleId}")
    @Operation(summary = "Attach previously uploaded evidence to an owned after-sales application")
    public CommonResult<Void> attach(@PathVariable Long afterSaleId,
                                     @Valid @RequestBody AfterSaleEvidence.AttachRequest request) {
        evidenceService.attach(afterSaleId, request);
        return CommonResult.success(null);
    }

    @GetMapping("/{afterSaleId}/download")
    @Operation(summary = "Create a short-lived download URL for owned after-sales evidence")
    public CommonResult<AfterSaleEvidence.DownloadResponse> download(@PathVariable Long afterSaleId,
            @RequestParam @Pattern(regexp = "^after-sales/[1-9][0-9]*/[0-9a-f-]{36}\\.(jpg|png|pdf)$") String objectKey) {
        return CommonResult.success(evidenceService.download(afterSaleId, objectKey));
    }
}
