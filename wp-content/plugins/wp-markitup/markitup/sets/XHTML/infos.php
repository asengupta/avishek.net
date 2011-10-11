<?php
	if (!class_exists('XHTMLTagSet', false)) {
		class XHTMLTagSet extends TagSet {
		
			function getTagSetName() {
				return 'XHTML';
			}	
		}
	}
	$tagset = new XHTMLTagSet();
?>